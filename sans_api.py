import cv2
import math
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Plage HSV pour la couleur de peau
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        max_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(max_contour) > 3000:
            # Enveloppe convexe (retourne les indices des points)
            hull_indices = cv2.convexHull(max_contour, returnPoints=False)

            if len(hull_indices) > 3:
                # Extraire les défauts de convexité
                defects = cv2.convexityDefects(max_contour, hull_indices)
                valid_defects = 0

                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i].flatten()
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])

                        # Distances des côtés du triangle
                        a = math.sqrt(
                            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
                        )
                        b = math.sqrt(
                            (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2
                        )
                        c = math.sqrt(
                            (far[0] - end[0]) ** 2 + (far[1] - end[1]) ** 2
                        )

                        # Angle au creux (théorème d'Al-Kashi)
                        angle = math.acos(
                            (b**2 + c**2 - a**2) / (2 * b * c)
                        ) * (180 / math.pi)

                        # Filtre : angle aigu et profondeur minimale
                        depth = d / 256.0
                        if angle <= 90 and depth > 20:
                            valid_defects += 1
                            cv2.circle(
                                frame, far, 5, (0, 0, 255), -1
                            )  # Creux en rouge

                # N creux valides = N + 1 doigts levés
                finger_count = valid_defects + 1 if valid_defects > 0 else 0

                cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Doigts: {finger_count}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

    cv2.imshow("Comptage de doigts", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()