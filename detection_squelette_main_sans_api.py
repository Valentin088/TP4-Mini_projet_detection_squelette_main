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

# IDs des extrémités des 4 doigts (Index, Majeur, Annulaire, Auriculaire)
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
                # Pour la main gauche à l'écran, le pouce ouvert est vers la droite (X_bout > X_articulation)
                if handLms.landmark[4].x > handLms.landmark[3].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            else:
                # Pour la main droite à l'écran, le pouce ouvert est vers la gauche (X_bout < X_articulation)
                if handLms.landmark[4].x < handLms.landmark[3].x:
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

    # Calcul et affichage du nombre d'images par seconde (FPS)
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
