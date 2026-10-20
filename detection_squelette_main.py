import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Masque les avertissements d'initialisation de TensorFlow / MediaPipe dans le terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==============================================================================
# 1. INITIALISATION DE MEDIAPIPE (API Tasks)
# ==============================================================================

# Spécifie le fichier du modèle pré-entraîné
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

# Configuration des options du détecteur
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,                       # Détection de 2 mains maximum
    min_hand_detection_confidence=0.7, # Seuil de confiance pour détecter une main
    min_tracking_confidence=0.5        # Seuil de confiance pour le suivi
)

# Création du détecteur de main
detector = vision.HandLandmarker.create_from_options(options)

# IDs des extrémités des 4 doigts principaux (Index: 8, Majeur: 12, Annulaire: 16, Auriculaire: 20)
finger_tips = [8, 12, 16, 20]

# ==============================================================================
# 2. FLUX VIDÉO WEBCAM
# ==============================================================================

# Initialisation de la webcam (0 = caméra par défaut)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Erreur de lecture de la webcam.")
        break

    # Application de l'effet miroir (inversion horizontale)
    frame = cv2.flip(frame, 1)

    # Conversion de l'image de BGR (OpenCV) vers RGB (MediaPipe)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Conversion du format NumPy vers le format Image propre à MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Lancement de la détection sur l'image
    results = detector.detect(mp_image)

    total_fingers = 0

    # ==========================================================================
    # 3. ANALYSE ET COMPTAGE DES DOIGTS
    # ==========================================================================

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            fingers_up = []

            # ------------------------------------------------------------------
            # A. POUCE (Analyse sur l'axe X)
            # ------------------------------------------------------------------
            # Récupération de la latéralité de la main ("Left" ou "Right")
            hand_label = results.handedness[results.hand_landmarks.index(hand_landmarks)][0].category_name

            # Avec le cv2.flip, la main gauche réelle s'affiche à gauche de l'écran :
            if hand_label == "Left":
                # Main gauche à l'écran : pouce ouvert vers la gauche (X du bout < X de l'articulation)
                if hand_landmarks[4].x < hand_landmarks[3].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            else:
                # Main droite à l'écran : pouce ouvert vers la droite (X du bout > X de l'articulation)
                if hand_landmarks[4].x > hand_landmarks[3].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            # ------------------------------------------------------------------
            # B. AUTRES DOIGTS (Analyse sur l'axe Y)
            # ------------------------------------------------------------------
            # L'axe Y va de 0.0 (haut de l'image) à 1.0 (bas de l'image).
            # Un doigt est levé si la pointe (tip) est plus haute que l'articulation (tip - 2).
            for tip in finger_tips:
                if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            # Ajout des doigts levés de cette main au total
            total_fingers += fingers_up.count(1)

            # ------------------------------------------------------------------
            # C. DESSIN DU SQUELETTE DE LA MAIN
            # ------------------------------------------------------------------
            for lm in hand_landmarks:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    # ==========================================================================
    # 4. AFFICHAGE RENDER
    # ==========================================================================

    # Affichage du score total à l'écran
    cv2.putText(frame, f'Doigts: {total_fingers}', (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Affichage de la fenêtre OpenCV
    cv2.imshow("Tracking de la main & Detection des doigts", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()