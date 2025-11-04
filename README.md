# CPS843 Final Project — Speed Limit Detection 🚗💨

This project detects and reads speed limit signs using OpenCV and Tesseract OCR.

## Overview
The program automatically locates digits within an image of a speed sign and uses OCR to recognize the speed value. It supports rectangular (North American) and orange construction zone signs.

### Example: Put original Image file Name
Input:  
<img src="data/speed_scene.jpg" width="300"> 

Output: Will Highlight speed state Maximum Speed 
<img src="results/speed_limit_result_robust.png" width="300"> 

Result: **MAX 50**

## ⚙️ Requirements
- Python 3.9+
- OpenCV
- NumPy
- Pillow
- pytesseract
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

Install dependencies:
```bash
pip install opencv-python numpy pillow pytesseract
