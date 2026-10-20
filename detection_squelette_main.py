import os
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Masque les avertissements d'initialisation dans le terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==============================================================================
# 1. INITIALISATION DE MEDIAPIPE & CONNEXIONS DU SQUELETTE
# ==============================================================================

# Liste des paires de points à relier pour former le squelette de la main
HAND_CONNECTIONS = [
    # Paume / Poignet
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pouce
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Majeur
    (9, 13), (13, 14), (14, 15), (15, 16),  # Annulaire
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Auriculaire & Bas de la paume
]

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

finger_tips = [8, 12, 16, 20]

# Variables pour le calcul des FPS
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
            for tip in finger_tips:
                if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            total_fingers += fingers_up.count(1)

            # ------------------------------------------------------------------
            # B. DESSIN DU SQUELETTE (Traits de liaison + Points)
            # ------------------------------------------------------------------
            h, w, _ = frame.shape

            # 1. Tracé des lignes (traits)
            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = hand_landmarks[start_idx]
                p2 = hand_landmarks[end_idx]
                
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # 2. Tracé des points par-dessus les lignes
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    # ==========================================================================
    # 3. CALCUL ET AFFICHAGE DES FPS & INFOS
    # ==========================================================================

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Affichage du compteur FPS (en magenta) et des Doigts (en vert)
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 40), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.putText(frame, f'Doigts: {total_fingers}', (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Tracking de la main & Detection des doigts", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()