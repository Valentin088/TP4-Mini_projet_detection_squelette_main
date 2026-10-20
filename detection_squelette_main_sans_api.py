import cv2
import mediapipe as mp
import time
import math
import numpy as np

# 1. Initialisation de la webcam (0 = caméra intégrée/par défaut)
cap = cv2.VideoCapture(0)

# 2. Initialisation des objets MediaPipe
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

Hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks des extrémités des doigts (Pouce: 4, Index: 8, Majeur: 12, Annulaire: 16, Auriculaire: 20)
ALL_FINGER_TIPS = [4, 8, 12, 16, 20]
finger_tips = [8, 12, 16, 20]

# Variables pour le calcul des FPS
PreviousTime = 0
CurrentTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Impossible d'accéder à la webcam.")
        break

    # Application du miroir horizontal
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    # MediaPipe nécessite une image en format RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Traitement de l'image et détection des repères
    result = Hands.process(imgRGB)

    total_fingers = 0

    # Si au moins une main est détectée
    if result.multi_hand_landmarks and result.multi_handedness:
        for handLms, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            fingers_up = []
            
            # Récupération de la latéralité (Gaucher / Droitier)
            hand_label = handedness.classification[0].label

            # ------------------------------------------------------------------
            # A. COMPTAGE DU POUCE (Axe X) - Ajusté avec l'effet miroir
            # ------------------------------------------------------------------
            # Avec cv2.flip(), l'étiquette "Left" correspond visuellement à votre main droite et inversement
            if hand_label == "Left":
                # Logique inversée pour compenser le miroir horizontal
                if handLms.landmark[4].x < handLms.landmark[3].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            else:
                if handLms.landmark[4].x > handLms.landmark[3].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            # ------------------------------------------------------------------
            # B. COMPTAGE DES 4 AUTRES DOIGTS (Axe Y)
            # ------------------------------------------------------------------
            for tip in finger_tips:
                if handLms.landmark[tip].y < handLms.landmark[tip - 2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            total_fingers += fingers_up.count(1)

            # ------------------------------------------------------------------
            # C. TRACÉ DU SQUELETTE
            # ------------------------------------------------------------------
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Cercle violet sur le poignet (landmark 0) comme dans le tutoriel
            cx0, cy0 = int(handLms.landmark[0].x * w), int(handLms.landmark[0].y * h)
            cv2.circle(img, (cx0, cy0), 15, (255, 0, 255), cv2.FILLED)

            # ------------------------------------------------------------------
            # D. EFFET INTERACTIF ENTRE DOIGTS PROCHES
            # ------------------------------------------------------------------
            THRESHOLD_DIST = 100  # Seuil de déclenchement en pixels

            # Coordonnées réelles en pixels de chaque bout de doigt
            tip_coords = [(int(handLms.landmark[idx].x * w), int(handLms.landmark[idx].y * h)) for idx in ALL_FINGER_TIPS]

            # Comparaison de toutes les paires de doigts de la main
            for i in range(len(tip_coords)):
                for j in range(i + 1, len(tip_coords)):
                    ptA = tip_coords[i]
                    ptB = tip_coords[j]

                    # Calcul de la distance euclidienne
                    dist = math.hypot(ptA[0] - ptB[0], ptA[1] - ptB[1])

                    if dist < THRESHOLD_DIST:
                        ratio = max(0.0, min(1.0, dist / THRESHOLD_DIST))

                        # L'épaisseur du trait et la taille du point augmentent à l'approche
                        thickness = int(np.interp(dist, [10, THRESHOLD_DIST], [8, 2]))
                        center_radius = int(np.interp(dist, [10, THRESHOLD_DIST], [16, 4]))

                        # Transition de couleur (BGR) : Vert -> Jaune -> Rouge
                        green = int(255 * ratio)
                        red = int(255 * (1.0 - ratio))
                        color = (0, green, red)

                        # Centre du segment
                        mid_x = (ptA[0] + ptB[0]) // 2
                        mid_y = (ptA[1] + ptB[1]) // 2

                        # Tracé de la liaison et du point milieu
                        cv2.line(img, ptA, ptB, color, thickness)
                        cv2.circle(img, (mid_x, mid_y), center_radius, color, cv2.FILLED)
                        cv2.circle(img, (mid_x, mid_y), center_radius + 2, (255, 255, 255), 1)

    # --------------------------------------------------------------------------
    # AFFICHAGE DES INFOS & FPS
    # --------------------------------------------------------------------------
    CurrentTime = time.time()
    fps = 1 / (CurrentTime - PreviousTime) if (CurrentTime - PreviousTime) > 0 else 0
    PreviousTime = CurrentTime

    # Textes superposés à la vidéo
    cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.putText(img, f'Doigts: {total_fingers}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Hand Tracking MediaPipe", img)

    # Quitter la boucle avec la touche 'd' (comme dans le tutoriel) ou 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d') or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
