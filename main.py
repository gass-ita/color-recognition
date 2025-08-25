
# Import OpenCV for image processing and NumPy for numerical operations
import cv2
import numpy as np


"""
# =======  COLOR CALIBRATION  =======
# This block (commented out) describes a calibration procedure for color correction.
# It captures reference colors from a color chart using the webcam, computes a correction matrix,
# and saves it for future use. The matrix maps camera colors to standard RGB values.
"""
    

# Use identity matrix for color correction (no-op) unless calibration is performed
M = np.eye(3, dtype=np.float32)  # Identity matrix for now


# ======= COLOR DEFINITIONS =======
# List of colors to detect, with HSV thresholds and drawing colors
COLORS = [
    {
        "name": "Red",  # Color name
        "hsv_lower": np.array([0, 180, 150], np.uint8),   # Lower HSV bound for red
        "hsv_upper": np.array([10, 255, 255], np.uint8),  # Upper HSV bound for red
        "rgb_pure":   (0, 0, 255),      # Pure red in BGR for drawing
        "irreg_color": (0, 165, 255),   # Orange for irregular shapes
    },
    {
        "name": "Green",
        "hsv_lower": np.array([40, 150, 120], np.uint8),
        "hsv_upper": np.array([85, 255, 255], np.uint8),
        "rgb_pure":   (0, 255, 0),      # Pure green in BGR
        "irreg_color": (0, 255, 255)    # Yellow for irregular shapes
    },
    {
        "name": "Blue",
        "hsv_lower": np.array([95, 180, 120], np.uint8),
        "hsv_upper": np.array([130, 255, 255], np.uint8),
        "rgb_pure":   (255, 0, 0),      # Pure blue in BGR
        "irreg_color": (255, 127, 0)    # Light blue/orange for irregular shapes
    }, 
    {
        "name": "Yellow",
        "hsv_lower": np.array([20, 150, 150], np.uint8),
        "hsv_upper": np.array([30, 255, 255], np.uint8),
        "rgb_pure":   (0, 255, 255),      # Pure yellow in BGR
        "irreg_color": (0, 255, 0)        # Green for irregular shapes
    }
]


# --------- Helper Functions ----------
def hsv_accuracy(mask, hsv_img, center_hsv):
    """
    Calculate the mean HSV value and accuracy of a masked region compared to a reference HSV value.
    Returns mean HSV and an accuracy score (0-100).
    """
    hsv_masked = hsv_img[mask != 0]
    if hsv_masked.size == 0:
        return (np.zeros(3), 0.0)
    mean_hsv = hsv_masked.mean(axis=0)
    dist = np.linalg.norm(mean_hsv.astype(np.float32) - center_hsv.astype(np.float32))
    accuracy = max(0, 1 - dist / 80) * 100
    return mean_hsv, accuracy

def adjust_gamma(image, gamma=1.0):
    """
    Apply gamma correction to an image to adjust brightness/contrast.
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


# ----------- MAIN LOOP ---------------
GAMMA_VALUE = 0.4  # Gamma correction value
cap = cv2.VideoCapture(0)  # Open webcam

# DUMMY COLOR CORRECTION (identity here), replace with your real 3x3 matrix
M = np.eye(3)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # --- Color Correction (currently identity matrix, so no change) ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # Convert to RGB and normalize
    h, w, c = frame_rgb.shape
    corrected = frame_rgb.reshape(-1,3) @ M  # Apply color correction matrix
    corrected = np.clip(corrected, 0, 1)     # Ensure values are in [0,1]
    corrected_img = (corrected.reshape(h, w, 3) * 255).astype(np.uint8)  # Convert back to uint8
    corrected_bgr = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR)       # Convert back to BGR for OpenCV

    # --- Gamma Correction ---
    gamma_img = adjust_gamma(corrected_bgr, gamma=GAMMA_VALUE)

    # --- HSV CONVERSION ---
    hsvFrame = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2HSV)  # Convert to HSV for color segmentation
    vis_img = gamma_img.copy()                             # Image for drawing detections
    pure_map = np.zeros_like(vis_img)                      # Image for pure color mask

    kernel = np.ones((5,5), "uint8")                      # Kernel for mask dilation

    # Loop through each color definition
    for color in COLORS:
        # Create mask for color using HSV thresholds
        mask = cv2.inRange(hsvFrame, color["hsv_lower"], color["hsv_upper"])
        mask = cv2.dilate(mask, kernel)  # Dilate mask to fill gaps
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
        center_hsv = (color["hsv_lower"] + color["hsv_upper"]) / 2  # Reference HSV value
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # Ignore small regions
                min_rect = cv2.minAreaRect(contour)  # Minimum area rectangle
                rect_w, rect_h = min_rect[1]
                rect_area = rect_w * rect_h
                extent_rot = area / rect_area if rect_area != 0 else 0  # How much contour fills rectangle
                roi_mask = np.zeros_like(mask)
                cv2.drawContours(roi_mask, [contour], -1, 255, -1)  # Mask for current contour
                mean_hsv, accuracy = hsv_accuracy(roi_mask, hsvFrame, center_hsv)  # Get mean HSV and accuracy
                rectish = extent_rot > 0.8  # Is the shape rectangular?
                color_bgr = color["rgb_pure"] if rectish else color["irreg_color"]  # Choose color for drawing
                if rectish:
                    # Draw rectangle and label for regular shapes
                    label = "{} ({:.0f}%) Rect".format(color["name"], accuracy)
                    box = cv2.boxPoints(min_rect)
                    box = np.intp(box)
                    cv2.drawContours(vis_img, [box], 0, color_bgr, 2)
                    # For pure_map: only rect-ish regions
                    mask_rect = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(mask_rect, [contour], -1, 255, -1)
                    pure_map[mask_rect != 0] = color["rgb_pure"]
                else:
                    # Draw bounding box and label for irregular shapes
                    x, y, w2, h2 = cv2.boundingRect(contour)
                    label = "{} ({:.0f}%) Irreg".format(color["name"], accuracy)
                    cv2.rectangle(vis_img, (x, y), (x+w2, y+h2), color_bgr, 2)
                # Use point from contour or bounding rect for label position
                x, y = tuple(contour[0][0]) if rectish else (x, y)
                cv2.putText(vis_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)

    # Stack images for display: gamma-corrected, detection, and pure mask
    stacked = np.hstack([gamma_img, vis_img, pure_map])
    cv2.imshow("Color corrected | Detected (Rect/Irreg, acc%) | Pure mask", stacked)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()