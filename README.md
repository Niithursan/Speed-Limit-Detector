# Speed Limit Detector

**Robust Speed Limit Detection in Unconstrained Environments using Hybrid Computer Vision and OCR**

> **Course:** CPS843 - Computer Vision
> **Language:** Python, OpenCV, Tesseract

---

## Overview

This project implements a lightweight, explainable Computer Vision pipeline to detect and read North American speed limit signs. Unlike "Black Box" Deep Learning models (like YOLO), this system uses classical computer vision techniques—geometric logic, adaptive thresholding, and histogram equalization—to achieve robust performance on standard CPUs without requiring a GPU.

It features a custom **Digit Stitching Algorithm** to solve OCR segmentation errors on highway signs (e.g., reading "1 0 0" as "100").

---

## Key Features

* **Hybrid Detection Pipeline:**
    * **Primary Strategy (Adaptive Contours):** Uses Gaussian Adaptive Thresholding to find the rectangular border of signs, cropping them for high-precision OCR.
    * **Fallback Strategy (Global Scan):** Automatically switches to a full-image sparse text scan if the sign is occluded or the border is damaged.
* **Smart Preprocessing:** Uses **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and Bicubic Upscaling (2.0x) to make signs readable in deep shadows or bright sunlight.
* **Digit Stitching Logic:** A custom algorithm that merges fragmented digits based on vertical alignment and horizontal proximity (Dynamic Tolerance), fixing common OCR errors on widely-spaced highway signs.
* **Validation Layer:** Filters results against a hash set of valid North American speed limits to eliminate false positives (e.g., house numbers).

---

## Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/speed-limit-detector.git](https://github.com/yourusername/speed-limit-detector.git)
cd speed-limit-detector
```

### 2. Install Python Dependencies
```bash
pip install opencv-python numpy pytesseract
```

### 3. Install Tesseract OCR Engine
You must have the Tesseract binary installed on your system.
* **Windows:** [Download Installer](https://github.com/UB-Mannheim/tesseract/wiki)
    * *Note:* Add Tesseract to your System PATH or update the path in `batch_process.py`.
* **Mac:** `brew install tesseract`
* **Linux:** `sudo apt install tesseract-ocr`

---

## How to Run

1.  **Add Images:** Place your test images (`.jpg`, `.png`) in the `data/` folder.
2.  **Execute:** Run the main batch processor:
    ```bash
    python batch_process.py
    ```
3.  **View Results:**
    * **Visual Output:** Check `results_batch/` for images annotated with bounding boxes and speed labels.
    * **Data Report:** Open `experiment_results.csv` for a detailed log including detected speed, confidence score, and the strategy used (Contour vs. Global).

---

## The Pipeline

The system processes images in three distinct stages:

### Stage 1: Preprocessing
The image is converted to grayscale and enhanced using **CLAHE** to balance lighting. It is then upscaled by **2.0x** to give the OCR engine more pixels to work with.

### Stage 2: Hybrid Detection
The system attempts **Strategy A** first. It looks for rectangular contours with specific aspect ratios ($0.5 < AR < 1.3$).
* *Success:* It crops the sign and runs OCR on just that small rectangle.
* *Failure:* If no rectangle is found (e.g., sign is hidden by a tree), it triggers **Strategy B**, scanning the entire image for text.

### Stage 3: Digit Stitching & Validation
Raw OCR output often splits numbers (e.g., `["1", "0", "0"]`). The **Stitching Algorithm** analyzes the geometry:
> *Are these numbers on the same vertical line? Is the gap between them small relative to their height?*

If yes, it merges them into `"100"` before validating the final number against standard limits.

---

## Project Structure

```text
.
├── batch_process.py       # Main application logic (Pipeline & Stitching)
├── data/                  # Input directory for raw images
├── results_batch/         # Output directory for annotated images
├── experiment_results.csv # Generated performance report (Auto-generated)
└── README.md              # Project Documentation
```

---

## Results

| Condition | Success Rate | Notes |
| :--- | :--- | :--- |
| **High Contrast (Sunny)** | 100% | Perfect detection via Adaptive Contours. |
| **Urban Clutter** | 100% | Geometric filtering successfully ignored non-speed signs. |
| **Highway Signs** | 100% | **Stitching Algorithm** successfully merged split digits. |
| **Overall Accuracy** | **92%** | Tested on diverse dataset including occlusion and shadows. |
