import cv2
import numpy as np
import pytesseract
import re
from pathlib import Path

# --- CONFIGURATION ---
# Set your Tesseract path here if needed (Windows example):
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Standard North American Speed Limits (mph and km/h)
VALID_SPEEDS = {
    10, 15, 20, 25, 30, 35, 40, 45, 50, 
    55, 60, 65, 70, 75, 80, 85, 90, 
    100, 110, 120, 130
}

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
    # Edge detection optimized for sign borders
    edged = cv2.Canny(blur, 50, 150) 
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    h_img, w_img = img_bgr.shape[:2]
    
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # We look for rectangles (4 points) with reasonable size
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            
            # Aspect ratio filter: Speed signs are taller than wide (AR < 1.0) 
            # or slightly square. US signs are ~0.75 to 0.9.
            if 0.5 < ar < 1.1 and w > 50 and h > 50:
                roi = img_bgr[y:y+h, x:x+w]
                rois.append((roi, (x,y,w,h)))
                
    return rois

def ocr_image(crop, psm=7):
    """Run OCR on a specific cropped image."""
    # Preprocess the crop: grayscale -> upscale -> threshold
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = get_contrasted(gray)
    
    # Try two different thresholds to be robust against lighting
    thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 31, 2)
    
    results = []
    cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
    
    for th in [thresh1, thresh2]:
        data = pytesseract.image_to_data(th, config=cfg, output_type=pytesseract.Output.DICT)
        # Parse results
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text.isdigit():
                conf = int(data['conf'][i])
                val = int(text)
                # CRITICAL: Validate against known speed limits
                if val in VALID_SPEEDS and conf > 40:
                    results.append((val, conf))
                    
    # Return best result by confidence
    results.sort(key=lambda x: x[1], reverse=True)
    return results[0] if results else None

# --- MAIN PIPELINE ---
def scan_speed_limit(img_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read {img_path}")
        return

    print(f"Processing {img_path}...")
    
    # Strategy 1: Try to find the sign rectangle first
    candidates = isolate_rects(img)
    
    best_speed = None
    best_conf = 0
    final_bbox = None # (x,y,w,h)

    # Check all detected rectangles
    for roi, bbox in candidates:
        res = ocr_image(roi)
        if res:
            speed, conf = res
            print(f"  -> Candidate ROI found: {speed} (Conf: {conf})")
            if conf > best_conf:
                best_speed = speed
                best_conf = conf
                final_bbox = bbox

    # Strategy 2: Fallback (Global Scan)
    if best_speed is None:
        print("  -> No sign rectangle detected. Scanning full image...")
        res = ocr_image(img, psm=11) 
        if res:
            best_speed, best_conf = res
            # Set bbox to the whole image
            final_bbox = (0, 0, img.shape[1], img.shape[0])

    # --- VISUALIZATION (FIXED) ---
    vis = img.copy()
    
    if best_speed:
        # Draw the main box
        if final_bbox:
            x, y, w, h = final_bbox
            color = (0, 255, 0) # Green
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 4)

            # --- LABEL POSITIONING FIX ---
            label = f"SPEED: {best_speed}"
            
            # Get text size
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            
            # Calculate label position
            # If the box is at the very top (y < 40), draw label INSIDE the box
            if y < 40:
                text_y = y + th + 20 
                bg_y1 = y
                bg_y2 = y + th + 30
            else:
                # Otherwise draw ABOVE the box
                text_y = y - 10
                bg_y1 = y - th - 20
                bg_y2 = y

            # Draw background for text (Black box for contrast)
            cv2.rectangle(vis, (x, bg_y1), (x + tw + 20, bg_y2), color, -1)
            
            # Draw text (Black or White)
            cv2.putText(vis, label, (x + 10, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    else:
        # Failed case
        label = "NO LIMIT FOUND"
        cv2.putText(vis, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imwrite(out_path, vis)
    print(f"Saved result to {out_path} | Detected: {best_speed}")

# Example usage
scan_speed_limit("data/25.jpeg", "results/25Result.png")