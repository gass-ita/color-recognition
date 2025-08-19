# 🎨 Real-time Color Detection & Calibration with OpenCV

This project captures live video from a webcam and detects **red, green,
and blue objects** in real time. It applies **color correction, gamma
correction, and HSV-based segmentation**, then classifies detected
shapes as **rectangular** or **irregular**. A **pure-color map** is also
generated to highlight detected patches.

------------------------------------------------------------------------

## 📌 Features

-   **Color Calibration**
    -   Uses a 5-patch reference chart (Red, Green, Blue, White,
        Black).\
    -   Computes a **3×3 correction matrix** to align captured colors
        with reference values.\
    -   Saves/loads calibration data (`color_correction_matrix.npy`).\
    -   *(Currently set to identity matrix, calibration code commented
        out).*
-   **Gamma Correction**
    -   Adjustable gamma for brightness/contrast tuning (`GAMMA_VALUE`).
-   **Color Detection (HSV Masks)**
    -   Detects **Red, Green, and Blue** objects using HSV thresholds.\
    -   Red is handled with dual ranges (0--10° and 170--180° hue).\
    -   Morphological **dilation** reduces noise.
-   **Shape Classification**
    -   Finds contours and calculates rotated bounding rectangles.\
    -   If filled area ≈ rectangle area → classified as **Rect**.\
    -   Otherwise → **Irregular**.\
    -   Displays detection **accuracy (%)** based on HSV match.
-   **Visualization**
    -   **Left panel** → Color-corrected & gamma-adjusted frame.\
    -   **Middle panel** → Detected objects (labeled Rect/Irreg with
        accuracy).\
    -   **Right panel** → Pure color map showing only valid rectangular
        patches.

------------------------------------------------------------------------

## 🖼️ Example Output

When running, three windows are combined side by side:

1.  **Color corrected + gamma adjusted frame**\
2.  **Detected objects with bounding boxes + accuracy labels**\
3.  **Pure color pixel map (only clean red/green/blue rectangles)**

------------------------------------------------------------------------

## 🚀 Usage

### Requirements

-   Python 3.7+
-   OpenCV (`cv2`)
-   NumPy

Install dependencies:

``` bash
pip install opencv-python numpy
```

### Run

``` bash
python main.py
```

Press **`q`** to exit the live window.

------------------------------------------------------------------------

## ⚙️ Configuration

-   **Calibration**:\
    Uncomment the calibration block at the top to compute your own
    correction matrix.\
    Show a chart with R, G, B, W, K patches and click them in order when
    prompted.

-   **Gamma**:\
    Adjust `GAMMA_VALUE` (default `0.4`) for your lighting conditions.

-   **HSV thresholds**:\
    Defined in the script under `# ======= HSV MASKS =======`.\
    Adjust if your camera or screen colors differ.


------------------------------------------------------------------------

## 🎯 Notes

-   Works best under consistent lighting.\
-   Designed for detecting **solid colored rectangles** (e.g., screen
    test patterns).\
-   Irregular shapes will still be detected but labeled differently.\
-   Accuracy (%) is relative to the HSV distance from the target color
    center.
