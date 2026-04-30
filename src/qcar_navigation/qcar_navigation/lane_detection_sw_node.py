#!/usr/bin/env python3
"""
=======================================================================
 Lane Detection Node — QCar Navigation | Sudo Drive
 Authors: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 SW + Poly2 v6 — UNA SOLA LÍNEA AMARILLA CENTRAL

 Diseñado para:
   - Una cinta amarilla central que divide dos carriles
   - Interior con luz variable (ventanas, sol directo)
   - El QCar sigue la línea central manteniéndola centrada en el frame

 Cambio fundamental vs versiones anteriores:
   NO se buscan dos líneas L y R para calcular un centro.
   Se detecta UNA SOLA línea amarilla y se publica su posición.
   El error lateral = posición_línea - centro_imagen

 Pipeline:
   BGR → CLAHE → HSV → máscara amarillo
   → ROI → histograma → UN SOLO conjunto de ventanas
   → polyfit de la línea central → publicar centroide = posición línea

 Published Topics:
   /amh19/lane/debug_image   (sensor_msgs/Image)
   /amh19/lane/lines         (Float32MultiArray) [a, b, c] del poly
   /amh19/lane/centroid      (geometry_msgs/Point) posición de la línea
=======================================================================
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image

_QOS_LATEST = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

DEFAULT_IN_TOPIC       = '/amh19/undistorted/csi_front'
DEFAULT_DEBUG_TOPIC    = '/amh19/lane/debug_image'
DEFAULT_LINES_TOPIC    = '/amh19/lane/lines'
DEFAULT_CENTROID_TOPIC = '/amh19/lane/centroid'

CLR_LINE   = (0, 220, 255)
CLR_CENTER = (0, 255, 255)
CLR_MEM    = (180, 180, 0)


class LaneDetectionNode(Node):

    def __init__(self):
        super().__init__('lane_detection_node')

        # ── Topics ────────────────────────────────────────────────────
        self.declare_parameter('in_topic',       DEFAULT_IN_TOPIC)
        self.declare_parameter('debug_topic',    DEFAULT_DEBUG_TOPIC)
        self.declare_parameter('lines_topic',    DEFAULT_LINES_TOPIC)
        self.declare_parameter('centroid_topic', DEFAULT_CENTROID_TOPIC)

        # ── ROI vertical ──────────────────────────────────────────────
        self.declare_parameter('roi_top',    0.45)
        self.declare_parameter('roi_bottom', 0.97)

        # ── ROI horizontal (trapecio) ─────────────────────────────────
        self.declare_parameter('roi_bl', 0.02)
        self.declare_parameter('roi_br', 0.98)
        self.declare_parameter('roi_tl', 0.20)
        self.declare_parameter('roi_tr', 0.80)

        # ── Sliding Window (una sola línea) ───────────────────────────
        self.declare_parameter('n_windows', 12)
        self.declare_parameter('win_width', 40)
        self.declare_parameter('min_pix',   20)

        # ── CLAHE ─────────────────────────────────────────────────────
        self.declare_parameter('clahe_clip', 2.0)
        self.declare_parameter('clahe_grid',   8)

        # ── Binarización HSV ──────────────────────────────────────────
        self.declare_parameter('hue_low',   15)
        self.declare_parameter('hue_high',  38)
        self.declare_parameter('sat_low',   40)
        self.declare_parameter('val_low',   60)

        # ── Estabilidad ───────────────────────────────────────────────
        self.declare_parameter('max_age',      15)
        self.declare_parameter('smooth_alpha', 0.30)
        self.declare_parameter('max_curv', 0.050)

        # ── Debug ──────────────────────────────────────────────────────
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('show_windows',  True)

        p = lambda n: self.get_parameter(n).value
        self.in_topic       = p('in_topic')
        self.debug_topic    = p('debug_topic')
        self.lines_topic    = p('lines_topic')
        self.centroid_topic = p('centroid_topic')
        self.roi_top        = p('roi_top')
        self.roi_bottom     = p('roi_bottom')
        self.roi_bl         = p('roi_bl')
        self.roi_br         = p('roi_br')
        self.roi_tl         = p('roi_tl')
        self.roi_tr         = p('roi_tr')
        self.n_windows      = p('n_windows')
        self.win_width      = p('win_width')
        self.min_pix        = p('min_pix')
        self.clahe_clip     = p('clahe_clip')
        self.clahe_grid     = p('clahe_grid')
        self.hue_low        = p('hue_low')
        self.hue_high       = p('hue_high')
        self.sat_low        = p('sat_low')
        self.val_low        = p('val_low')
        self.max_age        = p('max_age')
        self.alpha          = p('smooth_alpha')
        self.max_curv       = p('max_curv')
        self.publish_debug  = p('publish_debug')
        self.show_windows   = p('show_windows')

        # ── Estado interno ─────────────────────────────────────────────
        self.bridge   = CvBridge()
        self.n_frames = 0
        self._poly    = None
        self._last    = None
        self._age     = 0
        self._curv    = 0.0
        self._mode    = 'init'

        # ── Objetos cacheados (evitar recracion cada frame) ────────────
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_grid, self.clahe_grid))
        self._k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self._hsv_lower = np.array(
            [self.hue_low, self.sat_low, self.val_low], np.uint8)
        self._hsv_upper = np.array(
            [self.hue_high, 255, 255], np.uint8)

        # ROI mask cacheada: se invalida si cambia el tamaño del frame
        self._roi_mask   = None
        self._roi_shape  = (0, 0)

        # Hilo único para debug — no bloquea el pipeline de control
        self._debug_pool = ThreadPoolExecutor(max_workers=1)
        self._debug_busy = False
        self._debug_lock = threading.Lock()

        self.sub = self.create_subscription(
            Image, self.in_topic, self.cb, _QOS_LATEST)
        self.pub_debug    = self.create_publisher(
            Image, self.debug_topic, _QOS_LATEST)
        self.pub_lines    = self.create_publisher(
            Float32MultiArray, self.lines_topic, 10)
        self.pub_centroid = self.create_publisher(
            Point, self.centroid_topic, 10)

        g = self.get_logger().info
        g('=' * 62)
        g(' LANE DETECTION NODE  [v6 — línea central única]')
        g('=' * 62)
        g(f'  CLAHE  clip={self.clahe_clip}  grid={self.clahe_grid}x{self.clahe_grid}')
        g(f'  HSV    H:[{self.hue_low},{self.hue_high}]  S>={self.sat_low}  V>={self.val_low}')
        g(f'  ROI    base [{self.roi_bl:.2f}..{self.roi_br:.2f}]  top [{self.roi_tl:.2f}..{self.roi_tr:.2f}]')
        g(f'  SW     n={self.n_windows}  w=±{self.win_width}  min={self.min_pix}')
        g(f'  max_curv={self.max_curv}  max_age={self.max_age}')
        g('  Debug rendering: hilo separado (no bloquea control)')
        g('=' * 62)

    # ═══════════════════════════════════════════════════════════════════
    #  CALLBACK
    # ═══════════════════════════════════════════════════════════════════
    def cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(str(e))
            return
        if frame is None or frame.size == 0:
            return

        h, w = frame.shape[:2]
        self.n_frames += 1

        binary   = self._binarize(frame)
        roi_mask = self._build_roi(h, w)
        binary   = cv2.bitwise_and(binary, binary, mask=roi_mask)

        line_pts, win_boxes, base_x = self._sliding_window(binary, h, w)
        poly = self._fit_poly(line_pts)
        poly = self._smooth(poly)
        cx, cy = self._centroid(poly, h, w)

        # Publicar control inmediatamente — sin esperar al debug
        self._pub_lines(poly)
        self._pub_centroid(cx, cy)

        if self.publish_debug:
            # Si el hilo de debug ya está ocupado, descartamos este frame
            # de debug (no afecta control, solo visualización)
            with self._debug_lock:
                if self._debug_busy:
                    return
                self._debug_busy = True

            # Copiar solo lo necesario para el hilo
            frame_cp = frame.copy()
            bin_cp   = binary.copy()
            poly_cp  = poly.copy() if poly is not None else None
            pts_cp   = line_pts.copy() if len(line_pts) else line_pts
            age      = self._age
            curv     = self._curv
            nf       = self.n_frames
            header   = msg.header

            self._debug_pool.submit(
                self._draw_and_publish,
                frame_cp, bin_cp, roi_mask, win_boxes,
                pts_cp, poly_cp, cx, cy, base_x,
                h, w, age, curv, nf, header)

    def _draw_and_publish(self, frame, binary, roi_mask, win_boxes,
                          pts, poly, cx, cy, base_x,
                          h, w, age, curv, n_frames, header):
        try:
            dbg = self._draw(frame, binary, roi_mask, win_boxes,
                             pts, poly, cx, cy, base_x,
                             h, w, age, curv, n_frames)
            out = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
            out.header = header
            self.pub_debug.publish(out)
        finally:
            with self._debug_lock:
                self._debug_busy = False

    # ═══════════════════════════════════════════════════════════════════
    #  BINARIZACIÓN — CLAHE + HSV AMARILLO
    # ═══════════════════════════════════════════════════════════════════
    def _binarize(self, frame):
        h_img, w_img = frame.shape[:2]

        # CLAHE en canal L (objeto cacheado — no se recrea cada frame)
        lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        frame_eq = cv2.cvtColor(
            cv2.merge([self._clahe.apply(l), a, b]),
            cv2.COLOR_LAB2BGR)

        hsv  = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # Kernels cacheados
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._k3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k5)

        n, lbls, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        if n <= 1:
            return np.zeros_like(mask)

        min_area = max(50,  int(w_img * h_img * 0.0005))
        max_area = int(w_img * h_img * 0.15)
        mid_y    = h_img * 0.55

        best_lbl   = -1
        best_score = -1

        for lbl in range(1, n):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue

            cy_blob = stats[lbl, cv2.CC_STAT_TOP] + stats[lbl, cv2.CC_STAT_HEIGHT] / 2
            if cy_blob < mid_y:
                continue

            bw = max(1, stats[lbl, cv2.CC_STAT_WIDTH])
            bh = max(1, stats[lbl, cv2.CC_STAT_HEIGHT])
            fill_ratio = area / (bw * bh)

            if fill_ratio > 0.85 and area > int(w_img * h_img * 0.03):
                continue

            score = area * (1.0 - cy_blob / h_img * 0.3)
            if score > best_score:
                best_score = score
                best_lbl   = lbl

        if best_lbl < 0:
            return np.zeros_like(mask)

        clean = np.zeros_like(mask)
        clean[lbls == best_lbl] = 255
        return clean

    # ═══════════════════════════════════════════════════════════════════
    #  ROI — cacheada, solo se reconstruye si cambia el tamaño del frame
    # ═══════════════════════════════════════════════════════════════════
    def _build_roi(self, h, w):
        if self._roi_shape == (h, w):
            return self._roi_mask

        mask  = np.zeros((h, w), np.uint8)
        y_top = int(h * self.roi_top)
        y_bot = int(h * self.roi_bottom)
        pts   = np.array([[
            (int(w * self.roi_bl), y_bot),
            (int(w * self.roi_tl), y_top),
            (int(w * self.roi_tr), y_top),
            (int(w * self.roi_br), y_bot),
        ]], np.int32)
        cv2.fillPoly(mask, pts, 255)
        self._roi_mask  = mask
        self._roi_shape = (h, w)
        return self._roi_mask

    # ═══════════════════════════════════════════════════════════════════
    #  SLIDING WINDOW — UNA SOLA LÍNEA
    # ═══════════════════════════════════════════════════════════════════
    def _sliding_window(self, binary, h, w):
        y_b  = int(h * self.roi_bottom)
        y_t  = int(h * self.roi_top)
        span = max(1, y_b - y_t)
        xl   = int(w * self.roi_bl)
        xr   = int(w * self.roi_br)

        z1   = binary[y_b - span//4 : y_b,           :]
        z2   = binary[y_b - span//2 : y_b - span//4, :]
        hist = (np.sum(z1, 0)*3 + np.sum(z2, 0)*2).astype(np.float32)
        hist[:xl] = 0; hist[xr:] = 0

        if self._poly is not None:
            base_x = int(np.clip(np.polyval(self._poly, y_b), xl, xr))
        else:
            base_x = int(np.argmax(hist)) if hist.max() > 0 else (xl+xr)//2

        curv  = self._curv
        aww   = self.win_width + int(min(curv * 30000, 25))
        win_h = max(1, span // self.n_windows)
        nz_y, nz_x = np.nonzero(binary)
        cur_x = base_x
        pts_idx, boxes = [], []

        for i in range(self.n_windows):
            yh = y_b - i * win_h
            yl = yh - win_h
            x1 = max(0, cur_x - aww)
            x2 = min(w, cur_x + aww)
            boxes.append(((x1, yl), (x2, yh)))

            good = np.where((nz_y >= yl) & (nz_y < yh) &
                            (nz_x >= x1) & (nz_x < x2))[0]
            pts_idx.extend(good)

            if len(good) >= self.min_pix:
                nc    = int(np.mean(nz_x[good]))
                cur_x = int(np.clip(nc, cur_x - aww*2, cur_x + aww*2))

        if not pts_idx:
            return np.empty((0,2), np.int32), boxes, base_x

        pts = np.column_stack((nz_x[pts_idx], nz_y[pts_idx]))
        return pts, boxes, base_x

    # ═══════════════════════════════════════════════════════════════════
    #  POLYFIT
    # ═══════════════════════════════════════════════════════════════════
    def _fit_poly(self, pts):
        if len(pts) < 3:
            return None
        try:
            c    = np.polyfit(pts[:,1], pts[:,0], 2)
            curv = abs(c[0])
            self._curv = curv  # siempre actualizar para HUD real
            if curv <= self.max_curv:
                return c
            else:
                self.get_logger().warn(
                    f'Poly rejected: curv={curv:.5f} > {self.max_curv:.5f}',
                    throttle_duration_sec=2.0)
                return None
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════════════
    #  SMOOTH + MEMORIA
    # ═══════════════════════════════════════════════════════════════════
    def _smooth(self, poly):
        if poly is not None:
            self._poly = poly if self._poly is None else \
                         self.alpha * poly + (1-self.alpha) * self._poly
            self._last = self._poly
            self._age  = 0
        else:
            self._age += 1
            self._poly = self._last if self._age <= self.max_age else None
        return self._poly

    # ═══════════════════════════════════════════════════════════════════
    #  CENTROIDE
    # ═══════════════════════════════════════════════════════════════════
    def _centroid(self, poly, h, w):
        y  = int(h * self.roi_bottom)
        cx = int(np.clip(np.polyval(poly, y), 0, w-1)) \
             if poly is not None else w // 2
        return cx, y

    # ═══════════════════════════════════════════════════════════════════
    #  PUBLISHERS
    # ═══════════════════════════════════════════════════════════════════
    def _pub_lines(self, poly):
        msg = Float32MultiArray()
        msg.data = list(poly.astype(float)) if poly is not None \
                   else [-1., -1., -1.]
        self.pub_lines.publish(msg)

    def _pub_centroid(self, cx, cy):
        msg = Point()
        msg.x, msg.y, msg.z = float(cx), float(cy), 0.
        self.pub_centroid.publish(msg)

    # ═══════════════════════════════════════════════════════════════════
    #  DEBUG — se ejecuta en hilo separado
    # ═══════════════════════════════════════════════════════════════════
    def _draw(self, frame, binary, roi_mask, win_boxes,
              pts, poly, cx, cy, base_x, h, w, age, curv, n_frames):
        dbg = frame.copy()

        # ROI overlay — operación vectorizada, sin array temporal extra
        roi_idx = roi_mask == 255
        dbg[roi_idx] = (dbg[roi_idx].astype(np.float32) * 0.65
                        + np.array([10, 15, 0], np.float32) * 0.35
                        ).astype(np.uint8)

        # Píxeles de la cinta — numpy directo, sin loop Python
        if len(pts):
            valid = ((pts[:,0] >= 0) & (pts[:,0] < w) &
                     (pts[:,1] >= 0) & (pts[:,1] < h))
            dbg[pts[valid, 1], pts[valid, 0]] = CLR_LINE

        # Ventanas SW
        if self.show_windows:
            for (p1, p2) in win_boxes:
                cv2.rectangle(dbg, p1, p2, (0, 180, 180), 1)

        # Curva polinomial
        y_range = np.linspace(int(h*self.roi_top),
                              int(h*self.roi_bottom), 100).astype(int)
        if poly is not None:
            xs  = np.polyval(poly, y_range).astype(int)
            pts_draw = np.array([[[x, y]] for x, y in zip(xs, y_range)
                                 if 0 <= x < w], np.int32)
            if len(pts_draw) >= 2:
                clr = CLR_LINE if age == 0 else CLR_MEM
                cv2.polylines(dbg, [pts_draw], False, clr, 4, cv2.LINE_AA)

        # Línea central de referencia (dashed)
        for y in range(0, h, 20):
            cv2.line(dbg, (w//2, y), (w//2, min(y+10,h)), (120,120,120), 1)

        cv2.circle(dbg, (cx, cy), 10, CLR_CENTER, -1)
        cv2.circle(dbg, (cx, cy), 14, CLR_CENTER, 2)
        cv2.arrowedLine(dbg, (w//2, cy), (cx, cy), CLR_CENTER, 2, tipLength=0.25)
        cv2.circle(dbg, (base_x, cy - 10), 5, (255, 100, 0), -1)

        # Inset binario
        ih, iw = h//4, w//4
        inset  = cv2.resize(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), (iw, ih))
        inset[inset[:,:,0] > 0] = [0, 200, 0]
        dbg[0:ih, w-iw:w] = inset
        cv2.rectangle(dbg, (w-iw, 0), (w, ih), (70,70,70), 1)
        cv2.putText(dbg, 'Binary', (w-iw+4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)

        # HUD
        err    = cx - w//2
        st     = 'OK' if age == 0 else \
                 (f'MEM({age})' if poly is not None else '---')
        clr_st = CLR_LINE if age == 0 else \
                 ((180,180,0) if poly is not None else (80,80,80))
        mode   = 'curve' if curv > 0.0003 else 'straight'
        mode_clr = (100,200,255) if mode == 'curve' else (150,255,150)

        cv2.rectangle(dbg, (0,0), (300, 88), (0,0,0), -1)
        cv2.putText(dbg, f'LINE  : {st}',              (8,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr_st, 1)
        cv2.putText(dbg, f'CX={cx}px  ERR={err:+d}px', (8,42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_CENTER, 1)
        cv2.putText(dbg, f'CURV={curv:.5f}  [{mode}]', (8,62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_clr, 1)
        cv2.putText(dbg, f'Frame #{n_frames}  [SW+Poly2 v6]', (8,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110,110,110), 1)

        return dbg

    def destroy_node(self):
        self._debug_pool.shutdown(wait=False)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
