import cv2
import numpy as np
import os

# ==== 1. Define Reference (Screen) RGB Patch Values ====
reference_rgbs = np.array([
    [255,   0,   0],  # Red
    [  0, 255,   0],  # Green
    [  0,   0, 255],  # Blue
    [255, 255, 255],  # White
    [  0,   0,   0],  # Black
], dtype=np.float32)

# ==== 2. Try to Load Correction Matrix ====
matrix_path = "color_correction_matrix.npy"
if os.path.exists(matrix_path):
    print("Loading correction matrix from file.")
    M = np.load(matrix_path)
else:
    print("No saved matrix found. Beginning calibration procedure.")
    # ==== 3. CALIBRATION PROCEDURE ====
    # (Make sure your screen displays the patches in any arrangement.)
    cap = cv2.VideoCapture(0)
    print("Press [SPACE] to grab a reference frame of your screen showing the color chart.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam failure. Check your camera.")
            exit()
        cv2.imshow("Reference Frame (press SPACE to capture)", frame)
        if cv2.waitKey(1) & 0xFF == 32:  # SPACE
            reference_capture = frame.copy()
            break
    cv2.destroyAllWindows()

    points = []
    captured_rgbs = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(reference_capture, (x, y), 8, (0, 255, 255), 2)
            cv2.imshow("Click each patch", reference_capture)
            patch = reference_capture[max(0, y-5):y+5, max(0, x-5):x+5]
            mean_color = np.mean(patch.reshape(-1,3), axis=0) # BGR
            rgb = mean_color[::-1]  # to RGB
            captured_rgbs.append(rgb)
            points.append((x,y))
            print(f"\nPatch #{len(points)} captured at ({x},{y}): RGB {rgb}")

    print(f"\nClick the center of each color patch ({len(reference_rgbs)} in total), in order matching the reference list.\n")
    print("Reference order:")
    for idx, rgb in enumerate(reference_rgbs):
        print(f"{idx+1}: {rgb}")
    cv2.imshow("Click each patch", reference_capture)
    cv2.setMouseCallback("Click each patch", on_mouse)

    # Click n patches
    while len(points) < len(reference_rgbs):
        if cv2.waitKey(10) == 27:
            print("\nCalibration cancelled by user.")
            exit()
    cv2.destroyAllWindows()

    captured_rgbs = np.array(captured_rgbs, dtype=np.float32)
    # Normalize both to 0-1
    reference_rgbs_norm = reference_rgbs / 255.0
    captured_rgbs_norm = captured_rgbs / 255.0

    # Least squares correction
    M, _, _, _ = np.linalg.lstsq(captured_rgbs_norm, reference_rgbs_norm, rcond=None)
    print("\nCorrection matrix computed:\n", M)

    np.save(matrix_path, M)
    cap.release()

# ==== 4. LIVE COLOR CORRECTION ====
print("\nStarting webcam with live color correction. Press ESC to exit.\n")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam failure.")
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w, c = frame_rgb.shape
    reshaped = frame_rgb.reshape(-1,3)
    corrected = reshaped @ M
    corrected = np.clip(corrected, 0, 1)
    corrected_img = (corrected.reshape(h,w,3) * 255).astype(np.uint8)
    corrected_bgr = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR)
    combined = np.hstack([frame, corrected_bgr])
    cv2.imshow("Original (LEFT) | Corrected (RIGHT)", combined)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()