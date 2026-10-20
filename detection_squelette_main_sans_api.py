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

# Variables pour le calcul des FPS
PreviousTime = 0
CurrentTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Impossible d'accéder à la webcam.")
        break

    # MediaPipe nécessite une image au format RGB (OpenCV utilise le format BGR)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Traitement de l'image et détection des repères
    result = Hands.process(imgRGB)

    # Si au moins une main est détectée
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # Récupération des coordonnées pour chaque point de la main
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Exemple : Dessine un cercle plus grand sur la base de la main (point ID 0)
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # Dessine les connexions (squelette) entre les points de la main
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calcul et affichage du nombre d'images par seconde (FPS)
    CurrentTime = time.time()
    fps = 1 / (CurrentTime - PreviousTime) if (CurrentTime - PreviousTime) > 0 else 0
    PreviousTime = CurrentTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    # Affichage de la vidéo dans une fenêtre
    cv2.imshow("Tracking de la main", img)

    # Quitter la boucle si l'utilisateur appuie sur la touche 'd'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources de la caméra et fermeture des fenêtres
cap.release()
cv2.destroyAllWindows()