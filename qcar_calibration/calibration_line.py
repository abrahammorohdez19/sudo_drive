#!/usr/bin/env python3
"""
=======================================================================
 HSV Calibrator — QCar Navigation | Sudo Drive
 Authors: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 Calibrador HSV interactivo offline.
 No necesita ROS2 corriendo — trabaja con imagen o video guardado.

 CÓMO CAPTURAR UNA IMAGEN DEL QCAR:
   # En tu laptop con ROS2 corriendo:
   ros2 run image_view image_saver --ros-args \
     -r image:=/amh19/undistorted/csi_front
   # Guarda 'left0000.jpg' en el directorio actual

   # O con ros2 topic:
   ros2 run image_view image_view --ros-args \
     -r image:=/amh19/undistorted/csi_front
   # Presiona 's' en la ventana para guardar

 USO:
   python3 hsv_calibrator.py imagen.jpg
   python3 hsv_calibrator.py video.mp4      # pausa con SPACE, samplea

 CONTROLES:
   Click izq   → samplea pixel (acumula hasta 50 samples)
   Click der   → borra todos los samples
   's'         → muestra el comando ROS2 listo para copiar/pegar
   'b'         → toggle vista: original / binario con umbrales actuales
   'r'         → reset samples
   SPACE       → pausa/resume (solo en video)
   '+' / '-'   → ajuste fino hue_low  (+1 / -1)
   ']' / '['   → ajuste fino hue_high (+1 / -1)
   'w' / 'q'   → ajuste fino sat_low  (+5 / -5)
   'e' / 'd'   → ajuste fino val_low  (+5 / -5)
   ESC         → salir
=======================================================================
"""

import sys
import cv2
import numpy as np


# ── Estado global ─────────────────────────────────────────────────────
samples     = []          # lista de (H, S, V) de clicks
hue_low     = 15
hue_high    = 35
sat_low     = 80
val_low     = 80
show_binary = False
paused      = False
current_frame = None


def mouse_callback(event, x, y, flags, param):
    global samples, hue_low, hue_high, sat_low, val_low
    if current_frame is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        hsv   = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        h_val = int(hsv[y, x, 0])
        s_val = int(hsv[y, x, 1])
        v_val = int(hsv[y, x, 2])
        bgr   = current_frame[y, x]
        samples.append((h_val, s_val, v_val))

        print(f'\n[SAMPLE #{len(samples)}]  pixel=({x},{y})')
        print(f'  BGR : B={bgr[0]}  G={bgr[1]}  R={bgr[2]}')
        print(f'  HSV : H={h_val}  S={s_val}  V={v_val}')

        if len(samples) >= 2:
            hs = [s[0] for s in samples]
            ss = [s[1] for s in samples]
            vs = [s[2] for s in samples]
            hue_low  = max(0,   min(hs) - 8)
            hue_high = min(180, max(hs) + 8)
            sat_low  = max(0,   min(ss) - 20)
            val_low  = max(0,   min(vs) - 20)
            print(f'\n  → Rango sugerido automático:')
            print(f'     hue_low={hue_low}  hue_high={hue_high}')
            print(f'     sat_low={sat_low}  val_low={val_low}')

    elif event == cv2.EVENT_RBUTTONDOWN:
        samples.clear()
        print('\n[RESET] Samples borrados')


def build_binary(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_low,  sat_low, val_low], np.uint8)
    upper = np.array([hue_high, 255,     255],      np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def draw_overlay(frame):
    """
    Dibuja sobre el frame:
      - Círculo rojo en cada sample
      - Panel de valores actuales
      - Preview de color amarillo detectado (esquina)
    """
    out = frame.copy()
    h_f, w_f = out.shape[:2]

    # Samples clickeados
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for i, (sh, ss, sv) in enumerate(samples):
        # Buscar posición aproximada del sample (no guardamos x,y — dibujamos leyenda)
        pass

    # Preview binario en esquina top-right
    binary = build_binary(frame)
    ih, iw = h_f // 4, w_f // 4
    bin_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    inset   = cv2.resize(bin_bgr, (iw, ih))

    # Colorear verde los píxeles activos del inset
    inset[inset[:,:,0] > 0] = [0, 220, 0]
    out[0:ih, w_f-iw:w_f] = inset
    cv2.rectangle(out, (w_f-iw, 0), (w_f, ih), (80,80,80), 1)
    cv2.putText(out, 'Binary preview', (w_f-iw+4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

    # Panel de valores
    panel_h = 160
    cv2.rectangle(out, (0, 0), (340, panel_h), (0,0,0), -1)

    lines = [
        (f'Samples: {len(samples)}  (click en la linea amarilla)',  (180,180,180)),
        (f'hue_low  = {hue_low:3d}   (+/-  teclas + / -)',          (80, 220, 220)),
        (f'hue_high = {hue_high:3d}   (+/-  teclas ] / [)',          (80, 220, 220)),
        (f'sat_low  = {sat_low:3d}   (+/-  teclas w / q)',           (180,180, 80)),
        (f'val_low  = {val_low:3d}   (+/-  teclas e / d)',           (180,100,180)),
        (f's=comando  b=binary  r=reset  ESC=salir',                 (120,120,120)),
    ]
    for i, (txt, clr) in enumerate(lines):
        cv2.putText(out, txt, (8, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1)

    # Color swatch del rango actual
    swatch = np.zeros((30, 80, 3), np.uint8)
    mid_h  = (hue_low + hue_high) // 2
    swatch[:] = cv2.cvtColor(
        np.array([[[mid_h, 200, 220]]], np.uint8),
        cv2.COLOR_HSV2BGR)[0][0]
    out[panel_h - 38 : panel_h - 8, 250:330] = swatch
    cv2.rectangle(out, (250, panel_h-38), (330, panel_h-8), (150,150,150), 1)
    cv2.putText(out, 'hue', (255, panel_h-42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)

    return out


def print_command():
    print('\n' + '='*60)
    print(' COMANDO ROS2 — copia y pega esto:')
    print('='*60)
    print(f'ros2 run <tu_paquete> lane_detection_node --ros-args \\')
    print(f'  -p hue_low:={hue_low}  \\')
    print(f'  -p hue_high:={hue_high}  \\')
    print(f'  -p sat_low:={sat_low}  \\')
    print(f'  -p val_low:={val_low}')
    print('='*60 + '\n')


def run_image(path):
    global current_frame, show_binary, hue_low, hue_high, sat_low, val_low

    frame = cv2.imread(path)
    if frame is None:
        print(f'ERROR: No se pudo cargar {path}')
        sys.exit(1)

    current_frame = frame.copy()
    win = 'HSV Calibrator'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 600)
    cv2.setMouseCallback(win, mouse_callback)

    print(f'\nImagen cargada: {path}  ({frame.shape[1]}x{frame.shape[0]})')
    print('Click en la línea amarilla para samplear. Presiona "s" para ver el comando.')

    while True:
        if show_binary:
            binary = build_binary(current_frame)
            display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            display[binary > 0] = [0, 220, 0]
            cv2.putText(display, f'BINARY  H:[{hue_low},{hue_high}] S>={sat_low} V>={val_low}',
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)
        else:
            display = draw_overlay(current_frame)

        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            print_command()
        elif key == ord('b'):
            show_binary = not show_binary
        elif key == ord('r'):
            samples.clear()
            print('[RESET] Samples borrados')
        elif key == ord('+') or key == ord('='):
            hue_low = min(hue_high - 1, hue_low + 1)
            print(f'hue_low → {hue_low}')
        elif key == ord('-'):
            hue_low = max(0, hue_low - 1)
            print(f'hue_low → {hue_low}')
        elif key == ord(']'):
            hue_high = min(180, hue_high + 1)
            print(f'hue_high → {hue_high}')
        elif key == ord('['):
            hue_high = max(hue_low + 1, hue_high - 1)
            print(f'hue_high → {hue_high}')
        elif key == ord('w'):
            sat_low = min(255, sat_low + 5)
            print(f'sat_low → {sat_low}')
        elif key == ord('q') or key == ord('Q'):
            sat_low = max(0, sat_low - 5)
            print(f'sat_low → {sat_low}')
        elif key == ord('e'):
            val_low = min(255, val_low + 5)
            print(f'val_low → {val_low}')
        elif key == ord('d'):
            val_low = max(0, val_low - 5)
            print(f'val_low → {val_low}')

    cv2.destroyAllWindows()


def run_video(path):
    global current_frame, show_binary, paused
    global hue_low, hue_high, sat_low, val_low

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f'ERROR: No se pudo abrir {path}')
        sys.exit(1)

    win = 'HSV Calibrator'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 600)
    cv2.setMouseCallback(win, mouse_callback)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f'\nVideo: {path}  ({total} frames @ {fps:.0f}fps)')
    print('SPACE=pausa  Click en la línea para samplear  s=comando')

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            current_frame = frame.copy()

        if current_frame is None:
            continue

        if show_binary:
            binary  = build_binary(current_frame)
            display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            display[binary > 0] = [0, 220, 0]
        else:
            display = draw_overlay(current_frame)

        fn = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.putText(display, f'frame {fn}/{total}  {"[PAUSA]" if paused else ""}',
                    (8, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

        cv2.imshow(win, display)
        wait = 0 if paused else max(1, int(1000 / fps))
        key  = cv2.waitKey(wait) & 0xFF

        if key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            print_command()
        elif key == ord('b'):
            show_binary = not show_binary
        elif key == ord('r'):
            samples.clear()
            print('[RESET]')
        elif key == ord('+') or key == ord('='):
            hue_low = min(hue_high - 1, hue_low + 1);  print(f'hue_low={hue_low}')
        elif key == ord('-'):
            hue_low = max(0, hue_low - 1);              print(f'hue_low={hue_low}')
        elif key == ord(']'):
            hue_high = min(180, hue_high + 1);          print(f'hue_high={hue_high}')
        elif key == ord('['):
            hue_high = max(hue_low+1, hue_high - 1);   print(f'hue_high={hue_high}')
        elif key == ord('w'):
            sat_low = min(255, sat_low + 5);            print(f'sat_low={sat_low}')
        elif key == ord('e'):
            val_low = min(255, val_low + 5);            print(f'val_low={val_low}')
        elif key == ord('d'):
            val_low = max(0, val_low - 5);              print(f'val_low={val_low}')

    cap.release()
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print('\nUSO:')
        print('  python3 hsv_calibrator.py imagen.jpg')
        print('  python3 hsv_calibrator.py video.mp4')
        print('\nPara capturar una imagen del QCar:')
        print('  ros2 run image_view image_saver --ros-args \\')
        print('    -r image:=/amh19/undistorted/csi_front')
        sys.exit(0)

    path = sys.argv[1]
    ext  = path.lower().split('.')[-1]

    if ext in ('jpg', 'jpeg', 'png', 'bmp', 'tiff'):
        run_image(path)
    elif ext in ('mp4', 'avi', 'mkv', 'mov', 'webm'):
        run_video(path)
    else:
        # Intenta como video primero, luego imagen
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            cap.release()
            run_video(path)
        else:
            run_image(path)

    print('\nValores finales:')
    print_command()


if __name__ == '__main__':
    main()