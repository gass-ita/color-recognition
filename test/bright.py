import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()
    detected_screens = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            rect_area = w * h
            extent = float(area) / rect_area
            if extent > 0.7 and 0.3 < w/float(h) < 1.7:
                detected_screens += 1

                # Draw rectangle
                cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)

                # Centroid position
                M = cv2.moments(approx)
                if M['m00'] > 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                else:
                    cx, cy = x + w//2, y + h//2

                # Label each detected screen
                cv2.putText(output, f"Screen {detected_screens}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.circle(output, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(output, f"({cx},{cy})", (cx-30, cy+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.putText(output, f"Detected screens: {detected_screens}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Multiple Screen Detection', output)
    cv2.imshow('Thresholded', thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()