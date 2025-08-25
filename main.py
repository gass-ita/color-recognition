import cv2
import numpy as np

"""
# =======  COLOR CALIBRATION  =======
REFERENCE_COLORS = np.array([
    [255,   0,   0],  # Red
    [  0, 255,   0],  # Green
    [  0,   0, 255],  # Blue
    [255, 255, 255],  # White
    [  0,   0,   0],  # Black
], dtype=np.float32)
MATRIX_FILE = "color_correction_matrix.npy"
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

# ======= COLOR DEFINITIONS =======
COLORS = [
    {
        "name": "Red",
        "hsv_lower": np.array([0, 180, 150], np.uint8),
        "hsv_upper": np.array([10, 255, 255], np.uint8),
        "rgb_pure":   (0, 0, 255),      # BGR for OpenCV drawing
        "irreg_color": (0, 165, 255),   # Orange for irregular
    },
    {
        "name": "Green",
        "hsv_lower": np.array([40, 150, 120], np.uint8),
        "hsv_upper": np.array([85, 255, 255], np.uint8),
        "rgb_pure":   (0, 255, 0),      # Green in BGR
        "irreg_color": (0, 255, 255)    # Yellow for irregular
    },
    {
        "name": "Blue",
        "hsv_lower": np.array([95, 180, 120], np.uint8),
        "hsv_upper": np.array([130, 255, 255], np.uint8),
        "rgb_pure":   (255, 0, 0),      # Blue in BGR
        "irreg_color": (255, 127, 0)    # Light blue/orange for irregular
    }, 
    {
        "name": "Yellow",
        "hsv_lower": np.array([20, 150, 150], np.uint8),
        "hsv_upper": np.array([30, 255, 255], np.uint8),
        "rgb_pure":   (0, 255, 255),      # Yellow in BGR
        "irreg_color": (0, 255, 0)        # Green for irregular
    }
]

# --------- Helper Function (for accuracy) ----------
def hsv_accuracy(mask, hsv_img, center_hsv):
    hsv_masked = hsv_img[mask != 0]
    if hsv_masked.size == 0:
        return (np.zeros(3), 0.0)
    mean_hsv = hsv_masked.mean(axis=0)
    dist = np.linalg.norm(mean_hsv.astype(np.float32) - center_hsv.astype(np.float32))
    accuracy = max(0, 1 - dist / 80) * 100
    return mean_hsv, accuracy

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# ----------- MAIN LOOP ---------------
GAMMA_VALUE = 0.4
cap = cv2.VideoCapture(0)

# DUMMY COLOR CORRECTION (identity here), replace with your real 3x3 matrix
M = np.eye(3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Color Correction (dummy here as identity) ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w, c = frame_rgb.shape
    corrected = frame_rgb.reshape(-1,3) @ M
    corrected = np.clip(corrected, 0, 1)
    corrected_img = (corrected.reshape(h, w, 3) * 255).astype(np.uint8)
    corrected_bgr = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR)

    # --- Gamma Correction ---
    gamma_img = adjust_gamma(corrected_bgr, gamma=GAMMA_VALUE)

    # --- HSV CONVERSION ---
    hsvFrame = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2HSV)
    vis_img = gamma_img.copy()
    pure_map = np.zeros_like(vis_img)

    kernel = np.ones((5,5), "uint8")

    for color in COLORS:
        mask = cv2.inRange(hsvFrame, color["hsv_lower"], color["hsv_upper"])
        mask = cv2.dilate(mask, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        center_hsv = (color["hsv_lower"] + color["hsv_upper"]) / 2
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                min_rect = cv2.minAreaRect(contour)
                rect_w, rect_h = min_rect[1]
                rect_area = rect_w * rect_h
                extent_rot = area / rect_area if rect_area != 0 else 0
                roi_mask = np.zeros_like(mask)
                cv2.drawContours(roi_mask, [contour], -1, 255, -1)
                mean_hsv, accuracy = hsv_accuracy(roi_mask, hsvFrame, center_hsv)
                rectish = extent_rot > 0.8
                color_bgr = color["rgb_pure"] if rectish else color["irreg_color"]
                if rectish:
                    label = "{} ({:.0f}%) Rect".format(color["name"], accuracy)
                    box = cv2.boxPoints(min_rect)
                    box = np.intp(box)
                    cv2.drawContours(vis_img, [box], 0, color_bgr, 2)
                    # For pure_map: only rect-ish
                    mask_rect = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(mask_rect, [contour], -1, 255, -1)
                    pure_map[mask_rect != 0] = color["rgb_pure"]
                else:
                    x, y, w2, h2 = cv2.boundingRect(contour)
                    label = "{} ({:.0f}%) Irreg".format(color["name"], accuracy)
                    cv2.rectangle(vis_img, (x, y), (x+w2, y+h2), color_bgr, 2)
                # Use point from contour or bounding rect for label position
                x, y = tuple(contour[0][0]) if rectish else (x, y)
                cv2.putText(vis_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)

    stacked = np.hstack([gamma_img, vis_img, pure_map])
    cv2.imshow("Color corrected | Detected (Rect/Irreg, acc%) | Pure mask", stacked)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()