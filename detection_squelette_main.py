import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Télécharge le modèle .task si tu ne l'as pas encore :
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Prépare les utilitaires de dessin
mp_hands = mp.solutions.hands if hasattr(mp, 'solutions') else None

finger_tips = [8, 12, 16, 20]
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Erreur de lecture de la webcam.")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Conversion de l'image OpenCV pour l'API Tasks
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = detector.detect(mp_image)

    total_fingers = 0

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            fingers_up = []

            # 1. Pouce (Axe X)
            # On récupère le type de main (Left / Right)
            hand_label = results.handedness[results.hand_landmarks.index(hand_landmarks)][0].category_name

            # Comme l'image est inversée avec cv2.flip, les directions sont interverties :
            if hand_label == "Left":
                # Pour la main gauche à l'écran, le pouce ouvert a un X plus petit que l'articulation
                if hand_landmarks[4].x < hand_landmarks[3].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            else:
                # Pour la main droite à l'écran, le pouce ouvert a un X plus grand
                if hand_landmarks[4].x > hand_landmarks[3].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            # 2. Les 4 autres doigts (Axe Y)
            for tip in finger_tips:
                if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            total_fingers += fingers_up.count(1)

            # Dessin simple des repères
            for lm in hand_landmarks:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    cv2.putText(frame, f'Doigts: {total_fingers}', (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Tracking de la main & Detection des doigts", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()