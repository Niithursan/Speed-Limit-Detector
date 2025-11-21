import os
import csv
import cv2
import numpy as np
import pytesseract
import re
from pathlib import Path

# --- CONFIGURATION ---
# This matches the folder structure in your screenshot
INPUT_FOLDER = "data"
OUTPUT_FOLDER = "results_batch"
CSV_REPORT = "experiment_results.csv"

# Valid speeds to filter out noise
VALID_SPEEDS = {
    10, 15, 20, 25, 30, 35, 40, 45, 50, 
    55, 60, 65, 70, 75, 80, 85, 90, 
    100, 110, 120, 130
}

# If Tesseract is not in your PATH, uncomment and fix this line:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- HELPER FUNCTIONS (The "Brain") ---

def get_contrasted(gray):
    """Apply CLAHE to maximize contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def isolate_rects(img_bgr):
    """
    Finds rectangular contours that look like speed signs.
    Returns a list of cropped images (ROIs).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150) 
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            
            # Check aspect ratio and size
            if 0.5 < ar < 1.1 and w > 30 and h > 30:
                roi = img_bgr[y:y+h, x:x+w]
                rois.append((roi, (x,y,w,h)))
                
    return rois

def ocr_image(crop, psm=7):
    """Run OCR on a specific cropped image."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = get_contrasted(gray)
    
    thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 31, 2)
    
    results = []
    cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
    
    for th in [thresh1, thresh2]:
        data = pytesseract.image_to_data(th, config=cfg, output_type=pytesseract.Output.DICT)
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text.isdigit():
                conf = int(data['conf'][i])
                val = int(text)
                if val in VALID_SPEEDS and conf > 40:
                    results.append((val, conf))
                    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[0] if results else None

def run_detection_on_single_image(img_path, out_path):
    """Main logic to process one image."""
    img = cv2.imread(img_path)
    if img is None: return None, 0

    candidates = isolate_rects(img)
    best_speed = None
    best_conf = 0
    final_bbox = None 

    # Strategy 1: Check crops
    for roi, bbox in candidates:
        res = ocr_image(roi)
        if res:
            speed, conf = res
            if conf > best_conf:
                best_speed = speed
                best_conf = conf
                final_bbox = bbox

    # Strategy 2: Check full image
    if best_speed is None:
        res = ocr_image(img, psm=11) 
        if res:
            best_speed, best_conf = res
            final_bbox = (0, 0, img.shape[1], img.shape[0])

    # Visualization
    vis = img.copy()
    if best_speed:
        x, y, w, h = final_bbox
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 4)
        
        label = f"SPEED: {best_speed}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        
        # Smart label positioning
        if y < 40:
            text_y = y + th + 20
            bg_y1, bg_y2 = y, y + th + 30
        else:
            text_y = y - 10
            bg_y1, bg_y2 = y - th - 20, y
            
        cv2.rectangle(vis, (x, bg_y1), (x + tw + 20, bg_y2), (0, 255, 0), -1)
        cv2.putText(vis, label, (x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    else:
        cv2.putText(vis, "FAIL", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imwrite(out_path, vis)
    return best_speed, best_conf

# --- BATCH PROCESSOR ---

def process_batch():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    with open(CSV_REPORT, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Detected Speed", "Confidence", "Status"])

        # Check if folder exists
        if not os.path.exists(INPUT_FOLDER):
            print(f"ERROR: The folder '{INPUT_FOLDER}' does not exist.")
            print(f"Current working directory: {os.getcwd()}")
            return

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]
        
        print(f"Found {len(files)} images in '{INPUT_FOLDER}'. Starting batch process...\n")

        total_processed = 0
        successful_detects = 0

        for filename in files:
            img_path = os.path.join(INPUT_FOLDER, filename)
            out_path = os.path.join(OUTPUT_FOLDER, "result_" + filename)
            
            detected_speed, conf = run_detection_on_single_image(img_path, out_path)
            
            status = "Success" if detected_speed else "Failed"
            writer.writerow([filename, detected_speed if detected_speed else "N/A", conf, status])
            
            if detected_speed:
                successful_detects += 1
            total_processed += 1
            print(f"Processed {filename}: {status} ({detected_speed})")

    print("\n" + "="*30)
    print(f"BATCH COMPLETE")
    print(f"Total Images: {total_processed}")
    print(f"Successful Detections: {successful_detects}")
    if total_processed > 0:
        print(f"Accuracy: {(successful_detects/total_processed)*100:.1f}%")
    print(f"Report saved to: {CSV_REPORT}")
    print("="*30)

if __name__ == "__main__":
    process_batch()