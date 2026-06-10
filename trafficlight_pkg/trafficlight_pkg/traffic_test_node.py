#!/usr/bin/env python3
"""
=======================================================================
 Trafficlight Detection Node — QCar Sudo Drive
 Author: Evan García Maldonado
 =======================================================================
Nodo ROS 2 unificado para el QCar optimizado para competencia:
- Filtra logs en terminal (evita spam de pista libre).
- Tópico simplificado: "Red", "Yellow", "Green", "Stop".
- Comandos XBee: "Break", "Slow down", "Continue", "Stop".
- Lógica de velocidad eliminada por completo.
"""

import os
import sys
import cv2
import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from ultralytics import YOLO

os.environ['ULTRALYTICS_OFFLINE'] = '1'
os.environ['YOLO_OFFLINE'] = '1'

class TrafficUnifiedNode(Node):
    def __init__(self):
        super().__init__('traffic_unified_node')

        # ==========================================
        # 1. PARÁMETROS CONFIGURABLES
        # ==========================================
        nuevo_model_path = '/home/evan/workspaces/quanser_acc_ws/src/compe/entrenamiento_yolo/YOYO.v1i.yolov8/best.pt'

        self.declare_parameter('model_path', nuevo_model_path)
        self.declare_parameter('confidence_threshold', 0.50)
        self.declare_parameter('xbee_port', '/dev/ttyUSB0')
        self.declare_parameter('xbee_enabled', True)

        # ==========================================
        # 2. CONFIGURACIÓN DE PUERTOS Y COMUNICACIÓN
        # ==========================================
        self.xbee_port = self.get_parameter('xbee_port').value
        self.xbee_enabled = bool(self.get_parameter('xbee_enabled').value)
        
        self._last_state = 'inicializando' 

        if self.xbee_enabled:
            self._init_xbee()

        # ==========================================
        # 3. INTERFACES ROS 2
        # ==========================================
        self.state_pub = self.create_publisher(String, '/compe/traffic_state', 10)
        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.get_logger().info('Creando suscripción al tópico de cámara del QCar...')
        self.image_sub = self.create_subscription(
            Image,
            '/qcar/decompressed/csi_front',
            self.image_callback,
            qos_profile
        )

        # ==========================================
        # 4. MODELO DE IA
        # ==========================================
        model_path = self.get_parameter('model_path').value
        if not os.path.exists(model_path):
            self.get_logger().error(f'Modelo no encontrado en: {model_path}')
            raise FileNotFoundError(model_path)

        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        self.prev_time = cv2.getTickCount()
        
        # Umbral intermedio para asegurar una detección estable sin ruido de fondo
        self.umbral_pixeles = 6  

        self.get_logger().info('=== Sistema Autónomo Listo (Rangos HSV Optimizados) ===')

    def _init_xbee(self) -> None:
        try:
            os.system(f'sudo chmod 666 {self.xbee_port}')
            os.system(f'stty -F {self.xbee_port} 9600 raw -echo')
            self.get_logger().info(f'XBee configurado en {self.xbee_port}')
        except Exception as e:
            self.get_logger().error(f'Fallo al inicializar hardware XBee: {e}')
            self.xbee_enabled = False

    def _send_xbee(self, state: str) -> None:
        if not self.xbee_enabled:
            return
        try:
            os.system(f"echo '{state}' > {self.xbee_port}")
        except Exception as e:
            self.get_logger().error(f'Error en transmisión serial: {e}')

    def classify_traffic_light_color(self, frame, x1, y1, x2, y2) -> str:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 'none'

        # Convertimos la Región de Interés (ROI) al espacio HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ---------------------------------------------------------------------
        # RANGOS AJUSTADOS: CORRECCIÓN DE AMARILLO VS ROJO (CON LUZ DE DÍA)
        # ---------------------------------------------------------------------
        
        # Rojo 1: Estrechado a 7 para dar margen al amarillo. Exige brillo e intensidad.
        low_red = np.array([0, 110, 150])
        upp_red = np.array([7, 255, 255])
        
        # Rojo 2: Extremo superior del matiz (Tonos más oscuros o purpúreos)
        low_red2 = np.array([170, 110, 150])
        upp_red2 = np.array([180, 255, 255])
        
        # Amarillo: SE BAJA A 10 para absorber los tonos naranja/ámbar que antes robaba el rojo
        low_yell = np.array([10, 90, 140])
        upp_yell = np.array([26, 255, 255]) 
        
        # Verde: Rango robusto y confiable aislado de los canales cálidos
        low_gren = np.array([30, 60, 140])
        upp_gren = np.array([100, 255, 255])

        # Generación de máscaras
        m_red = cv2.bitwise_or(cv2.inRange(hsv, low_red, upp_red), cv2.inRange(hsv, low_red2, upp_red2))
        m_yell = cv2.inRange(hsv, low_yell, upp_yell)
        m_gren = cv2.inRange(hsv, low_gren, upp_gren)

        # Conteo de píxeles activos por canal
        scores = {
            'Red': cv2.countNonZero(m_red),
            'Yellow': cv2.countNonZero(m_yell),
            'Green': cv2.countNonZero(m_gren),
        }

        # Determinamos cuál es el color predominante en la caja detectada por YOLO
        dominant = max(scores, key=scores.get)
        
        # Si ni el color dominante alcanza el mínimo de píxeles, se asume indefinido
        if scores[dominant] < self.umbral_pixeles:
            return 'none'

        return dominant

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            return

        results = self.model(frame, verbose=False)
        detected_label = "none"
        decision = "none"  
        best_conf = 0.0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if conf < self.get_parameter('confidence_threshold').value:
                    continue

                label = self.class_names[cls_id]
                label_lower = label.lower()

                if conf > best_conf:
                    best_conf = conf
                    detected_label = label

                # Prioridad al Stop Sign sobre el semáforo
                if 'stop' in label_lower:
                    decision = 'Stop'
                elif any(k in label_lower for k in ['traffic', 'semaforo', 'light', 'verde', 'rojo', 'amarillo']):
                    if decision != 'Stop':  
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        decision = self.classify_traffic_light_color(frame, x1, y1, x2, y2)

        # =========================================================================
        # MAQUINA DE ESTADOS MODIFICADA (SIN VELOCIDAD)
        # =========================================================================
        if decision != self._last_state:
            if decision == 'Stop':
                self._send_xbee("Stop")                     
                self.publicar_estado_string("Stop")          
                self.get_logger().error("[EVENTO] ALTO: Stop sign detectado.")

            elif decision == 'Red':
                self._send_xbee("Break")                    
                self.publicar_estado_string("Red")           
                self.get_logger().error("[EVENTO] ALTO: Semáforo Rojo detectado.")

            elif decision == 'Yellow':
                self._send_xbee("Slow down")                 
                self.publicar_estado_string("Yellow")        
                self.get_logger().warn("[EVENTO] PRECAUCIÓN: Semáforo Amarillo detectado.")

            elif decision == 'Green':
                self._send_xbee("Continue")                  
                self.publicar_estado_string("Green")         
                self.get_logger().info("[EVENTO] SIGA: Semáforo Verde detectado.")

            elif decision == 'none':
                self._send_xbee("Continue")
                self.publicar_estado_string("Green")

            self._last_state = decision

        # =========================================================================
        # INTERFAZ VISUAL EN PANTALLA
        # =========================================================================
        annotated_frame = results[0].plot()

        display_color = (0, 255, 0)
        display_text = "Clear Track"
        
        if self._last_state == 'Stop':
            display_text = "Stop Sign Detected"
            display_color = (0, 0, 255)
        elif self._last_state == 'Red':
            display_text = "Red Light"
            display_color = (0, 0, 255)
        elif self._last_state == 'Yellow':
            display_text = "Yellow Light"
            display_color = (0, 255, 255)
        elif self._last_state == 'Green':
            display_text = "Green Light"
            display_color = (0, 255, 0)

        cv2.putText(annotated_frame, f'YOLO: {detected_label} ({best_conf:.2f})', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Status: {display_text}', 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)
        
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - self.prev_time)
        self.prev_time = current_time
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow('Unified Navigation Control - QCar Subscription', annotated_frame)
        cv2.waitKey(1)

    def publicar_estado_string(self, texto_estado):
        msg = String()
        msg.data = texto_estado
        self.state_pub.publish(msg)

    def destroy_node(self):
        self.publicar_estado_string("Stop")
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TrafficUnifiedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()