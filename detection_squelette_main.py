import os
import cv2
import time
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Masque les avertissements d'initialisation dans le terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==============================================================================
# 1. INITIALISATION DE MEDIAPIPE & CONNEXIONS DU SQUELETTE
# ==============================================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pouce
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Majeur
    (9, 13), (13, 14), (14, 15), (15, 16),  # Annulaire
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Auriculaire & Paume
]

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Tous les bouts de doigts (Pouce: 4, Index: 8, Majeur: 12, Annulaire: 16, Auriculaire: 20)
ALL_FINGER_TIPS = [4, 8, 12, 16, 20]
finger_tips_no_thumb = [8, 12, 16, 20]

prev_time = 0

# ==============================================================================
# 2. FLUX VIDÉO WEBCAM
# ==============================================================================

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Erreur de lecture de la webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = detector.detect(mp_image)

    total_fingers = 0

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            fingers_up = []

            # ------------------------------------------------------------------
            # A. COMPTAGE DES DOIGTS
            # ------------------------------------------------------------------
            hand_label = results.handedness[results.hand_landmarks.index(hand_landmarks)][0].category_name

            # Pouce
            if hand_label == "Left":
                if hand_landmarks[4].x < hand_landmarks[3].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            else:
                if hand_landmarks[4].x > hand_landmarks[3].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            # Autres doigts
            for tip in finger_tips_no_thumb:
                if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            total_fingers += fingers_up.count(1)

            # ------------------------------------------------------------------
            # B. DESSIN DU SQUELETTE
            # ------------------------------------------------------------------
            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = hand_landmarks[start_idx]
                p2 = hand_landmarks[end_idx]
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

            # ------------------------------------------------------------------
            # C. DÉTECTION ET EFFET DE PROXIMITÉ ENTRE LES BOUTS DE DOIGTS
            # ------------------------------------------------------------------
            # Seuil de distance max en pixels pour déclencher le lien
            THRESHOLD_DIST = 60

            # Récupère les coordonnées pixel de toutes les extrémités
            tip_coords = [(int(hand_landmarks[idx].x * w), int(hand_landmarks[idx].y * h)) for idx in ALL_FINGER_TIPS]

            # Compare chaque paire de doigts unique
            for i in range(len(tip_coords)):
                for j in range(i + 1, len(tip_coords)):
                    ptA = tip_coords[i]
                    ptB = tip_coords[j]

                    # Calcul de la distance euclidienne en pixels
                    dist = math.hypot(ptA[0] - ptB[0], ptA[1] - ptB[1])

                    if dist < THRESHOLD_DIST:
                        # Ratio entre 0.0 (doigts collés) et 1.0 (doigts à la limite du seuil)
                        ratio = max(0.0, min(1.0, dist / THRESHOLD_DIST))

                        # 1. Épaisseur dynamique (plus proche = plus gros)
                        thickness = int(np.interp(dist, [10, THRESHOLD_DIST], [8, 2]))
                        center_radius = int(np.interp(dist, [10, THRESHOLD_DIST], [16, 4]))

                        # 2. Transition de couleur : Vert (255,0) -> Jaune -> Rouge (0,255)
                        # OpenCV utilise le format BGR : (Bleu, Vert, Rouge)
                        green = int(255 * ratio)
                        red = int(255 * (1.0 - ratio))
                        color = (0, green, red)

                        # Point central
                        mid_x = (ptA[0] + ptB[0]) // 2
                        mid_y = (ptA[1] + ptB[1]) // 2

                        # 3. Tracé du trait d'interaction et du point central
                        cv2.line(frame, ptA, ptB, color, thickness)
                        cv2.circle(frame, (mid_x, mid_y), center_radius, color, cv2.FILLED)
                        cv2.circle(frame, (mid_x, mid_y), center_radius + 2, (255, 255, 255), 1)

    # ==========================================================================
    # 3. STATS & AFFICHAGE
    # ==========================================================================

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f'FPS: {int(fps)}', (20, 40), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.putText(frame, f'Doigts: {total_fingers}', (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Tracking de la main & Detection des doigts", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()