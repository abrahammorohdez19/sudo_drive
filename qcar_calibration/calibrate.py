#!/usr/bin/env python3
"""
Calibración intrínseca de cámara del QCar1
Ejecutar DESPUÉS de collect_images.py
"""
import cv2
import numpy as np
import glob
import yaml

# ── CONFIGURACIÓN ─────────────────────────────────────
IMAGES_PATH  = '/home/nvidia/calibration/images/*.png'
OUTPUT_YAML  = '/home/nvidia/calibration/qcar_front.yaml'
CHECKERBOARD = (8, 5)       # esquinas interiores (no cuadros, sino esquinas)
SQUARE_SIZE  = 0.025        # metros — mide tu tablero impreso con regla
CAMERA_NAME  = 'csi_front'
# ──────────────────────────────────────────────────────

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def main():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],
                            0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints, imgpoints = [], []
    img_size = None

    images = sorted(glob.glob(IMAGES_PATH))
    print(f'\nProcesando {len(images)} imágenes...\n')

    for i, fname in enumerate(images):
        img  = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if found:
            corners_fine = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), CRITERIA)
            objpoints.append(objp)
            imgpoints.append(corners_fine)

            vis = img.copy()
            cv2.drawChessboardCorners(vis, CHECKERBOARD, corners_fine, found)
            cv2.imshow('Deteccion', vis)
            cv2.waitKey(300)
            print(f'  [OK] {i+1:02d}/{len(images)}: esquinas detectadas')
        else:
            print(f'  [--] {i+1:02d}/{len(images)}: tablero NO detectado — imagen descartada')

    cv2.destroyAllWindows()

    if len(objpoints) < 10:
        print(f'\n[ERROR] Solo {len(objpoints)} imágenes válidas. Necesitas mínimo 10.')
        print('Vuelve a capturar con mejor iluminación y más ángulos variados.')
        return

    print(f'\nCalibrando con {len(objpoints)} imágenes válidas...')
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    print(f'\n{"="*50}')
    print(f'  RMS reprojection error: {rms:.4f} px')
    if rms < 0.5:
        print('  Calidad: EXCELENTE ✓')
    elif rms < 1.0:
        print('  Calidad: BUENA ✓')
    else:
        print('  Calidad: BAJA — toma más fotos con más variedad de ángulos')
    print(f'{"="*50}')
    print(f'\n  fx = {K[0,0]:.2f}  fy = {K[1,1]:.2f}')
    print(f'  cx = {K[0,2]:.2f}  cy = {K[1,2]:.2f}')
    print(f'  distorsión: {D.flatten().round(5)}')

    # Guardar en formato ROS2
    w, h = img_size
    data = {
        'image_width': w, 'image_height': h,
        'camera_name': CAMERA_NAME,
        'distortion_model': 'plumb_bob',
        'camera_matrix':             {'rows':3,'cols':3,'data': K.flatten().tolist()},
        'distortion_coefficients':   {'rows':1,'cols':5,'data': D.flatten().tolist()},
        'rectification_matrix':      {'rows':3,'cols':3,'data': np.eye(3).flatten().tolist()},
        'projection_matrix':         {'rows':3,'cols':4,
                                      'data': np.hstack([K,np.zeros((3,1))]).flatten().tolist()}
    }
    with open(OUTPUT_YAML, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f'\n✓ YAML guardado en: {OUTPUT_YAML}')
    print('  Copia este archivo a tu paquete ROS2 en: config/qcar_front.yaml')

if __name__ == '__main__':
    main()