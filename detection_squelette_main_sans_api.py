import cv2
import mediapipe as mp
import time

# 1. Initialisation de la webcam (0 = caméra intégrée/par défaut)
cap = cv2.VideoCapture(0)

# 2. Initialisation des modules MediaPipe (API legacy)
mpHands = mp.solutions.hands
Hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

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

    # Application de l'effet miroir pour un comportement plus naturel
    img = cv2.flip(img, 1)

    # MediaPipe nécessite une image au format RGB (OpenCV utilise le format BGR)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Traitement de l'image et détection des repères
    result = Hands.process(imgRGB)

    total_fingers = 0

    # Si au moins une main est détectée
    if result.multi_hand_landmarks and result.multi_handedness:
        for handLms, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            fingers_up = []
            
            # Récupération du type de main (Left / Right)
            hand_label = handedness.classification[0].label

            # ------------------------------------------------------------------
            # A. Détection du Pouce (Axe X corrigé avec l'effet miroir)
            # ------------------------------------------------------------------
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
            # B. Détection des 4 autres doigts (Axe Y)
            # ------------------------------------------------------------------
            for tip in finger_tips:
                # Si le bout du doigt (ex: 8) est plus haut (Y plus petit) que l'articulation (ex: 6)
                if handLms.landmark[tip].y < handLms.landmark[tip - 2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            # Compte des doigts levés pour cette main
            total_fingers += fingers_up.count(1)

            # Dessine le squelette de la main
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calcul et affichage du nombre d'images par seconde (FPS)
    CurrentTime = time.time()
    fps = 1 / (CurrentTime - PreviousTime) if (CurrentTime - PreviousTime) > 0 else 0
    PreviousTime = CurrentTime

    # Affichage des informations sur la vidéo
    cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.putText(img, f'Doigts: {total_fingers}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Affichage de la vidéo dans une fenêtre
    cv2.imshow("Tracking de la main & Detection des doigts", img)

    # Quitter la boucle si l'utilisateur appuie sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources de la caméra et fermeture des fenêtres
cap.release()
cv2.destroyAllWindows()