# Speed Limit Detector 🚗💨

A robust Computer Vision project that detects and reads North American speed limit signs (US & Canada) using OpenCV and Tesseract OCR.

## 📝 Overview
This system automatically locates speed limit signs in street scenes and reads the speed value. It uses a **hybrid detection pipeline**:
1.  **Contour Isolation:** Finds the sign border and crops it for high-precision OCR.
2.  **Global Fallback:** Switches to a full-image scan if the sign border is blocked or unclear.
3.  **Validation:** Filters out noise by checking results against standard speed limits (e.g., 25, 45, 50, 100).

## ⚙️ Installation

1.  **Install Python Dependencies:**
    ```bash
    pip install opencv-python numpy pytesseract
    ```

2.  **Install Tesseract OCR:**
    * **Windows:** [Download Installer](https://github.com/UB-Mannheim/tesseract/wiki) (Make sure to check "Add to PATH" during installation).
    * **Mac:** `brew install tesseract`
    * **Linux:** `sudo apt install tesseract-ocr`

---

## How to Run

This will scan all images in your data folder and generate a report.

1.  **Prepare Images:**
    Put your test images (jpg, png) inside the `data/` folder.

2.  **Run the Script:**
    Open your terminal (or VS Code terminal) and run:
    ```bash
    python batch_process.py
    ```

3.  **Check Results:**
    * **Processed Images:** Go to the `results_batch/` folder to see the images with green boxes and speed labels.
    * **Data Report:** Open `experiment_results.csv` to see a table of all detected speeds and confidence scores.
---

## 📂 Project Structure
```text
.
├── batch_process.py       # The main script (Run this!)
├── data/                  # Folder where you put your input images
├── results_batch/         # Folder where results are saved
├── experiment_results.csv # Excel-compatible report of accuracy
└── README.md              # This file