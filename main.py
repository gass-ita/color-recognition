import cv2
import numpy as np

# =======  COLOR CALIBRATION  =======
REFERENCE_COLORS = np.array([
    [255,   0,   0],  # Red
    [  0, 255,   0],  # Green
    [  0,   0, 255],  # Blue
    [255, 255, 255],  # White
    [  0,   0,   0],  # Black
], dtype=np.float32)

"""MATRIX_FILE = "color_correction_matrix.npy"
if os.path.exists(MATRIX_FILE):
    print("Loading correction matrix from file.")
    M = np.load(MATRIX_FILE)
else:
    print("No saved matrix found. Beginning calibration procedure.")
    cap = cv2.VideoCapture(0)
    print("Show the 5 color chart patches (R, G, B, W, K) on screen.")
    print("Press [SPACE] to grab a frame, then click the center of each patch in order.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam failure.")
            exit()
        cv2.imshow("Reference Frame (SPACE to capture)", frame)
        if cv2.waitKey(1) & 0xFF == 32:
            reference_capture = frame.copy()
            break
    cv2.destroyAllWindows()

    points = []
    captured_rgbs = []
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(reference_capture, (x, y), 8, (0, 255, 255), 2)
            cv2.imshow("Click patches", reference_capture)
            patch = reference_capture[max(0, y-5):y+5, max(0, x-5):x+5]
            mean_color = np.mean(patch.reshape(-1,3), axis=0) # BGR
            rgb = mean_color[::-1]
            captured_rgbs.append(rgb)
            points.append((x, y))
            print(f"Patch #{len(points)} at ({x},{y}): RGB {rgb}")

    print("Click the patches: Red, Green, Blue, White, Black.")
    cv2.imshow("Click patches", reference_capture)
    cv2.setMouseCallback("Click patches", on_mouse)
    while len(points) < len(REFERENCE_COLORS):
        if cv2.waitKey(10) == 27:
            print("Calibration canceled.")
            exit()
    cv2.destroyAllWindows()
    captured_rgbs = np.array(captured_rgbs, dtype=np.float32)
    reference_rgbs_norm = REFERENCE_COLORS / 255.0
    captured_rgbs_norm = captured_rgbs / 255.0
    M, _, _, _ = np.linalg.lstsq(captured_rgbs_norm, reference_rgbs_norm, rcond=None)
    print("\nCorrection matrix computed:\n", M)
    np.save(MATRIX_FILE, M)
    cap.release()"""
    
M = np.eye(3, dtype=np.float32)  # Identity matrix for now

# =======  HELPERS  =======
def adjust_gamma(image, gamma=1.0):
    if abs(gamma-1.0) < 1e-6: return image
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def hsv_accuracy(mask, hsv_image, target_hsv):
    mean_hsv = cv2.mean(hsv_image, mask=mask)[:3]
    max_dist = np.linalg.norm(np.array([180,255,255]))
    dist = np.linalg.norm(np.array(mean_hsv) - np.array(target_hsv))
    raw_acc = 1.0 - (dist / max_dist)
    acc = max(0,min(1,raw_acc))
    return mean_hsv, acc*100

# =======  HSV MASKS (SCREEN OPTIMIZED)  =======
red_lower1 = np.array([0, 180, 150], np.uint8)
red_upper1 = np.array([10, 255, 255], np.uint8)
red_lower2 = np.array([170, 180, 150], np.uint8)
red_upper2 = np.array([180, 255, 255], np.uint8)
green_lower = np.array([40, 150, 120], np.uint8)
green_upper = np.array([85, 255, 255], np.uint8)
blue_lower = np.array([95, 180, 120], np.uint8)
blue_upper = np.array([130, 255, 255], np.uint8)

# =======  MAIN LOOP  =======
GAMMA_VALUE = 0.4

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- Color Correction --------
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w, c = frame_rgb.shape
    corrected = frame_rgb.reshape(-1,3) @ M
    corrected = np.clip(corrected, 0, 1)
    corrected_img = (corrected.reshape(h, w, 3) * 255).astype(np.uint8)
    corrected_bgr = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR)

    # -------- Gamma Correction --------
    gamma_img = adjust_gamma(corrected_bgr, gamma=GAMMA_VALUE)

    # -------- HSV masks --------
    hsvFrame = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsvFrame, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsvFrame, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    # Dilation
    kernel = np.ones((5,5), "uint8")
    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)

    vis_img = gamma_img.copy()  # for drawing on copy

    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center_red = ((red_lower1 + red_upper2) / 2)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            # Rotated rectangle extent check
            min_rect = cv2.minAreaRect(contour)
            rect_w, rect_h = min_rect[1]
            rect_area = rect_w * rect_h
            extent_rot = area / rect_area if rect_area != 0 else 0

            roi_mask = np.zeros_like(red_mask)
            cv2.drawContours(roi_mask, [contour], -1, 255, -1)
            mean_hsv, accuracy = hsv_accuracy(roi_mask, hsvFrame, center_red)
            rectish = extent_rot > 0.8

            if rectish:
                label = "Red ({:.0f}%) Rect".format(accuracy)
                color = (0, 0, 255)
                # Draw rotated rectangle instead of axis-aligned
                box = cv2.boxPoints(min_rect)
                box = np.intp(box)
                cv2.drawContours(vis_img, [box], 0, color, 2)
            else:
                label = "Red ({:.0f}%) Irreg".format(accuracy)
                color = (0,165,255)
                x, y, w2, h2 = cv2.boundingRect(contour)
                cv2.rectangle(vis_img, (x, y), (x + w2, y + h2), color, 2)
            cv2.putText(vis_img, label, tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ======= GREEN CONTOURS =======
    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center_green = (green_lower + green_upper) / 2
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            # Rotated rectangle extent check
            min_rect = cv2.minAreaRect(contour)
            rect_w, rect_h = min_rect[1]
            rect_area = rect_w * rect_h
            extent_rot = area / rect_area if rect_area != 0 else 0

            roi_mask = np.zeros_like(green_mask)
            cv2.drawContours(roi_mask, [contour], -1, 255, -1)
            mean_hsv, accuracy = hsv_accuracy(roi_mask, hsvFrame, center_green)
            rectish = extent_rot > 0.8

            if rectish:
                label = "Green ({:.0f}%) Rect".format(accuracy)
                color = (0,255,0)
                # Draw rotated rectangle instead of axis-aligned
                box = cv2.boxPoints(min_rect)
                box = np.intp(box)
                cv2.drawContours(vis_img, [box], 0, color, 2)
            else:
                label = "Green ({:.0f}%) Irreg".format(accuracy)
                color = (0,255,255)  # Yellow for irregular
                x, y, w2, h2 = cv2.boundingRect(contour)
                cv2.rectangle(vis_img, (x, y), (x + w2, y + h2), color, 2)
            cv2.putText(vis_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # ======= BLUE CONTOURS =======
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center_blue = (blue_lower + blue_upper) / 2
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            # Rotated rectangle extent check
            min_rect = cv2.minAreaRect(contour)
            rect_w, rect_h = min_rect[1]
            rect_area = rect_w * rect_h
            extent_rot = area / rect_area if rect_area != 0 else 0

            roi_mask = np.zeros_like(blue_mask)
            cv2.drawContours(roi_mask, [contour], -1, 255, -1)
            mean_hsv, accuracy = hsv_accuracy(roi_mask, hsvFrame, center_blue)
            rectish = extent_rot > 0.8

            if rectish:
                label = "Blue ({:.0f}%) Rect".format(accuracy)
                color = (255,0,0)
                # Draw rotated rectangle instead of axis-aligned
                box = cv2.boxPoints(min_rect)
                box = np.intp(box)
                cv2.drawContours(vis_img, [box], 0, color, 2)
            else:
                label = "Blue ({:.0f}%) Irreg".format(accuracy)
                color = (255,127,0)  # Light Blue for irregular
                x, y, w2, h2 = cv2.boundingRect(contour)
                cv2.rectangle(vis_img, (x, y), (x + w2, y + h2), color, 2)
            cv2.putText(vis_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ======= PURE COLOR PIXEL MAP =========
    pure_map = np.zeros_like(vis_img)

    # RED rect-ish only (rotated)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            min_rect = cv2.minAreaRect(contour)
            rect_w, rect_h = min_rect[1]
            rect_area = rect_w * rect_h
            extent_rot = area / rect_area if rect_area != 0 else 0
            rectish = extent_rot > 0.8
            if rectish and rect_w > 0 and rect_h > 0:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                pure_map[mask != 0] = [0, 0, 255]  # PURE RED

    # GREEN rect-ish only (rotated)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            min_rect = cv2.minAreaRect(contour)
            rect_w, rect_h = min_rect[1]
            rect_area = rect_w * rect_h
            extent_rot = area / rect_area if rect_area != 0 else 0
            rectish = extent_rot > 0.8
            if rectish and rect_w > 0 and rect_h > 0:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                pure_map[mask != 0] = [0, 255, 0]  # PURE GREEN

    # BLUE rect-ish only (rotated)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            min_rect = cv2.minAreaRect(contour)
            rect_w, rect_h = min_rect[1]
            rect_area = rect_w * rect_h
            extent_rot = area / rect_area if rect_area != 0 else 0
            rectish = extent_rot > 0.8
            if rectish and rect_w > 0 and rect_h > 0:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                pure_map[mask != 0] = [255, 0, 0]  # PURE BLUE

    stacked = np.hstack([gamma_img, vis_img, pure_map])
    cv2.imshow("Color corrected | Detected (Rect/Irreg, acc%) | Pure mask", stacked)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()