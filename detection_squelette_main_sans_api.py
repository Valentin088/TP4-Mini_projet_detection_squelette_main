import cv2
import numpy as np
import time
import math

# 1. Variables globales & paramètres de calibration
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None
hand_rect_two_x = None
hand_rect_two_y = None

is_calibrated = False
hand_hist = None
PreviousTime = 0

cap = cv2.VideoCapture(0)

# ==============================================================================
# A. DESSIN DES RECTANGLES DE ÉANTILLONNAGE
# ==============================================================================
def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y
    
    hand_rect_one_x = np.array(
        [6 * rows // 20, 6 * rows // 20, 6 * rows // 20, 
         9 * rows // 20, 9 * rows // 20, 9 * rows // 20, 
         12 * rows // 20, 12 * rows // 20, 12 * rows // 20], dtype=np.uint32)
    
    hand_rect_one_y = np.array(
        [9 * cols // 20, 10 * cols // 20, 11 * cols // 20, 
         9 * cols // 20, 10 * cols // 20, 11 * cols // 20, 
         9 * cols // 20, 10 * cols // 20, 11 * cols // 20], dtype=np.uint32)
    
    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)
    return frame

# ==============================================================================
# B. GÉNÉRATION DE L'HISTOGRAMME DE PEAU (HSV)
# ==============================================================================
def get_hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)
    
    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]
        
    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

# ==============================================================================
# C. SEGMENTATION PAR PROJECTION ARRIÈRE (Back Projection)
# ==============================================================================
def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cv2.filter2D(dst, -1, disc, dst)
    
    _, thresh = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)
    
    # Nettoyage morphologique pour effacer les bruits isolés
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
    
    return thresh

# ==============================================================================
# D. BOUCLE PRINCIPALE DE TRACKING
# ==============================================================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Erreur de lecture webcam.")
        break

    frame = cv2.flip(frame, 1)

    # 1. ÉTAPE DE CALIBRATION (Avant d'appuyer sur 'c')
    if not is_calibrated:
        frame = draw_rect(frame)
        cv2.putText(frame, "Place ta main dans les carrés et appuie sur 'c'", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 2. ÉTAPE DE TRACKING ET DÉTECTION (Après la calibration)
    else:
        mask = hist_masking(frame, hand_hist)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_fingers = 0
        finger_tips = []

        if contours:
            max_cont = max(contours, key=cv2.contourArea)

            if cv2.contourArea(max_cont) > 4000:
                epsilon = 0.012 * cv2.arcLength(max_cont, True)
                approx = cv2.approxPolyDP(max_cont, epsilon, True)

                # Centre géométrique de la paume (Transformée de Distance)
                dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                _, max_val, _, (cx, cy) = cv2.minMaxLoc(dist_transform)
                radius = int(max_val) if np.isfinite(max_val) else 0

                # Tracé du centre de la paume
                cv2.circle(frame, (cx, cy), max(radius, 5), (255, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                # Extraction des sommets (Bouts de doigts)
                hull_points = cv2.convexHull(approx, returnPoints=True)
                for pt in hull_points:
                    px, py = pt[0]
                    dist_to_center = math.hypot(px - cx, py - cy)

                    if dist_to_center > radius * 1.2 and py < cy + 30:
                        if all(math.hypot(px - ft[0], py - ft[1]) > 25 for ft in finger_tips):
                            finger_tips.append((px, py))

                # Dessin du squelette (Liaison Centre -> Doigts)
                for tip in finger_tips:
                    cv2.line(frame, (cx, cy), tip, (0, 255, 0), 2)
                    cv2.circle(frame, tip, 6, (0, 0, 255), -1)

                # Comptage des doigts via défauts de convexité
                hull_indices = cv2.convexHull(approx, returnPoints=False)
                if hull_indices is not None and len(hull_indices) > 3:
                    try:
                        defects = cv2.convexityDefects(approx, hull_indices)
                        if defects is not None:
                            for s, e, f, d in defects.reshape(-1, 4):
                                start = tuple(approx[s][0])
                                end = tuple(approx[e][0])
                                far = tuple(approx[f][0])

                                a = math.hypot(end[0] - start[0], end[1] - start[1])
                                b = math.hypot(far[0] - start[0], far[1] - start[1])
                                c = math.hypot(end[0] - far[0], end[1] - far[1])

                                if b * c != 0:
                                    angle = np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0))
                                    if angle <= np.pi / 2.2 and d > 7000:
                                        total_fingers += 1
                            if total_fingers > 0:
                                total_fingers += 1
                    except cv2.error:
                        pass

                # Interaction dynamique d'approche entre les doigts
                THRESHOLD_DIST = 70
                for i in range(len(finger_tips)):
                    for j in range(i + 1, len(finger_tips)):
                        ptA, ptB = finger_tips[i], finger_tips[j]
                        dist = math.hypot(ptA[0] - ptB[0], ptA[1] - ptB[1])

                        if dist < THRESHOLD_DIST:
                            ratio = max(0.0, min(1.0, dist / THRESHOLD_DIST))

                            thickness = int(np.interp(dist, [10, THRESHOLD_DIST], [8, 2]))
                            center_radius = int(np.interp(dist, [10, THRESHOLD_DIST], [14, 3]))

                            green = int(255 * ratio)
                            red = int(255 * (1.0 - ratio))
                            color = (0, green, red)

                            mid_x = (ptA[0] + ptB[0]) // 2
                            mid_y = (ptA[1] + ptB[1]) // 2

                            cv2.line(frame, ptA, ptB, color, thickness)
                            cv2.circle(frame, (mid_x, mid_y), center_radius, color, cv2.FILLED)
                            cv2.circle(frame, (mid_x, mid_y), center_radius + 2, (255, 255, 255), 1)

        cv2.putText(frame, f'Doigts: {total_fingers}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("Masque d'Histogramme", mask)

    # Calcul et affichage du compteur FPS
    CurrentTime = time.time()
    fps = 1 / (CurrentTime - PreviousTime) if (CurrentTime - PreviousTime) > 0 else 0
    PreviousTime = CurrentTime
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("Hand Tracking (Histogramme HSV)", frame)

    key = cv2.waitKey(1) & 0xFF
    # Appuie sur 'c' pour calibrer la couleur de ta main
    if key == ord('c'):
        hand_hist = get_hand_histogram(frame)
        is_calibrated = True
    # Appuie sur 'r' pour réinitialiser la calibration
    elif key == ord('r'):
        is_calibrated = False
    # Appuie sur 'q' ou 'd' pour quitter
    elif key == ord('q') or key == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()