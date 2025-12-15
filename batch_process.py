import os
import csv
import cv2
import numpy as np
import pytesseract

# --- CONFIGURATION ---
INPUT_FOLDER = "data"
OUTPUT_FOLDER = "results_batch"
CSV_REPORT = "experiment_results.csv"

# ROI filtering
ROI_MIN_SIZE = 30
ROI_AR_MIN = 0.5
ROI_AR_MAX = 1.3

# Morphology
DILATE_KERNEL_SIZE = (3, 3)
DILATE_ITERATIONS = 2

# OCR settings
PSM_ROI = 7
PSM_GLOBAL = 11
MIN_CONFIDENCE = 40

# Upscaling
UPSCALE_FACTOR = 2.0

# Valid speeds (North American standard)
VALID_SPEEDS = {
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    100,
    110,
    120,
    130,
}

# Uncomment if needed:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- HELPER FUNCTIONS ---


def get_contrasted(gray):
    """Apply CLAHE to maximize contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def isolate_rects(img_bgr):
    """
    Strategy 1: Find signs using Adaptive Thresholding (Best for white-on-sky).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Use Adaptive Thresholding instead of Canny (Better for invisible borders)
    # This turns the image into pure black/white blocks based on local contrast
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Dilate to connect broken edges (like the top of the sign)
    kernel = np.ones(DILATE_KERNEL_SIZE, np.uint8)
    thresh = cv2.dilate(
        thresh, kernel, iterations=DILATE_ITERATIONS
    )  # More aggressive dilation

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # Loose aspect ratio to catch signs viewed at an angle
        if ROI_AR_MIN < ar < ROI_AR_MAX and w > ROI_MIN_SIZE and h > ROI_MIN_SIZE:
            roi = img_bgr[y : y + h, x : x + w]
            rois.append((roi, (x, y, w, h)))

    return rois


def stitch_line(data):
    """
    CRITICAL RESTORATION: Merges separate digits (e.g., '8' and '0')
    if they are on the same line and close together.
    """
    merged_texts = []
    n = len(data["text"])

    i = 0
    while i < n:
        text = data["text"][i].strip()
        # Ensure confidence is an integer
        try:
            conf = int(data["conf"][i])
        except (ValueError, TypeError):
            conf = 0

        if text.isdigit() and conf > 30:
            current_num = text
            current_conf = conf
            
            # track index
            prev_idx = i

            # logic for look-ahead
            j = i + 1
            while j < n:
                next_text = data["text"][j].strip()
                try:
                    next_conf = int(data["conf"][j])
                except (ValueError, TypeError):
                    next_conf = 0

                # ---GEOMETRY CALCULATION---
                # 1. Vertical Alignment Check
                y_aligned = abs(data["top"][j] - data["top"][i]) < 10

                # 2. Horizontal Gap Check
                x_prev_end = data["left"][prev_idx] + data["width"][prev_idx]
                x_curr_start = data["left"][j]
                gap = x_curr_start - x_prev_end
                
                # Dynamic Tolerance
                # If the sign is huge (height=50px), we allow a gap of ~40px.
                # If the sign is small (height=10px), we only allow ~8px.
                char_height = data["height"][prev_idx]
                gap_allowed = gap < (char_height * 0.8) # 0.8 is a generous multiplier for highway signs

                # Check all conditions: Next is digit AND aligned vertically AND close horizontally
                if next_text.isdigit() and y_aligned and gap_allowed:
                    current_num += next_text
                    current_conf = (current_conf + next_conf) // 2  # Avg confidence
                    
                    # Update prev_idx so the NEXT digit is compared to THIS one
                    prev_idx = j 
                    j += 1
                else:
                    break

            merged_texts.append((current_num, current_conf))
            i = j  # Skip the consumed digits
        else:
            i += 1

    return merged_texts


def ocr_image(crop, psm=7):
    """Run OCR with Digit Stitching logic."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Preprocessing variants
    gray_lg = cv2.resize(
        gray, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC
    )

    variants = [
        get_contrasted(gray),  # Standard
        get_contrasted(gray_lg),  # Upscaled
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # Binary
    ]

    best_res = None
    cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"

    for img_var in variants:
        data = pytesseract.image_to_data(
            img_var, config=cfg, output_type=pytesseract.Output.DICT
        )

        # 1. Try standard parsing
        candidates = []
        for i in range(len(data["text"])):
            t = data["text"][i].strip()
            if t.isdigit():
                candidates.append((t, int(data["conf"][i])))

        # 2. Try STITCHED parsing (The Fix)
        stitched = stitch_line(data)
        candidates.extend(stitched)

        # Check all candidates
        for val_str, conf in candidates:
            try:
                val = int(val_str)
                if val in VALID_SPEEDS and conf > MIN_CONFIDENCE:
                    # Pick the one with highest confidence
                    if best_res is None or conf > best_res[1]:
                        best_res = (val, conf)
            except Exception:
                continue

    return best_res


def run_detection_on_single_image(img_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load {img_path}")
        return None, 0, "N/A"

    # Tracking which strategy succeeds
    strategy = "N/A"

    # Strategy 1: Adaptive Contours
    candidates = isolate_rects(img)
    best_speed = None
    best_conf = 0
    final_bbox = None

    for roi, bbox in candidates:
        res = ocr_image(roi, psm=PSM_ROI)  # Single block mode
        if res:
            speed, conf = res
            if conf > best_conf:
                best_speed = speed
                best_conf = conf
                final_bbox = bbox
                strategy = "Adaptive Contours"

    # Strategy 2: Global Scan (Fallback)
    if best_speed is None:
        print("  Falling back to global scan...")
        res = ocr_image(img, psm=PSM_GLOBAL)  # Sparse text mode
        if res:
            best_speed, best_conf = res
            final_bbox = (0, 0, img.shape[1], img.shape[0])
            strategy = "Global Scan (Fallback)"

    # Visualization
    vis = img.copy()
    if best_speed is not None:
        x, y, w, h = final_bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 4)

        label = f"SPEED: {best_speed}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        if y < 40:
            text_y = y + th + 20
            bg_y1, bg_y2 = y, y + th + 30
        else:
            text_y = y - 10
            bg_y1, bg_y2 = y - th - 20, y

        cv2.rectangle(vis, (x, bg_y1), (x + tw + 20, bg_y2), (0, 255, 0), -1)
        cv2.putText(
            vis, label, (x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
    else:
        cv2.putText(vis, "FAIL", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite(out_path, vis)
    return best_speed, best_conf, strategy


# --- BATCH PROCESSOR ---
def process_batch():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(CSV_REPORT, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Image Name", "Detected Speed", "Confidence", "Status", "Strategy"]
        )

        files = sorted(
            [
                f
                for f in os.listdir(INPUT_FOLDER)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        print(f"Found {len(files)} images. Processing...")

        for filename in files:
            img_path = os.path.join(INPUT_FOLDER, filename)
            out_path = os.path.join(OUTPUT_FOLDER, "result_" + filename)
            speed, conf, strategy = run_detection_on_single_image(img_path, out_path)

            status = "Success" if speed is not None else "Failed"
            writer.writerow(
                [
                    filename,
                    speed if speed is not None else "N/A",
                    conf,
                    status,
                    strategy,
                ]
            )
            print(f"Processed {filename}: {status} ({speed})")


if __name__ == "__main__":
    # Toggle between batch or single file here
    process_batch()
