import cv2, numpy as np, pytesseract, re
from pathlib import Path

# If Tesseract isn't on PATH (Windows), uncomment & set this:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMG = "data/110.jpg"
OUT = Path("results/speed_limit_result_robust.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

def mean_bgr_under(img, bbox):
    x, y, w, h = bbox
    x2, y2 = min(img.shape[1], x+w), min(img.shape[0], y+h)
    roi = img[max(0,y):y2, max(0,x):x2]
    if roi.size == 0:
        return (128,128,128)
    b,g,r = roi.reshape(-1,3).mean(axis=0)
    return (float(b), float(g), float(r))

def best_contrast_color(bgr):
    # Relative luminance (sRGB approximate)
    b,g,r = [c/255.0 for c in bgr]
    Y = 0.2126*r + 0.7152*g + 0.0722*b
    # choose bright color on dark background, dark color on bright background
    return (255,255,255) if Y < 0.45 else (0,0,0)  # white on dark, black on light

def draw_text_with_bg(img, text, org, font_scale=1.0, fg=(0,0,0), bg=(255,255,255), alpha=0.75):
    """Draw text with a rounded translucent background and a black outline for readability."""
    x, y = int(org[0]), int(org[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, font_scale, 2)
    pad = 8
    rect = (x-4, y-th-8, tw+pad, th+pad)  # (x, y, w, h) above the point

    # translucent background
    overlay = img.copy()
    cv2.rectangle(overlay, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    # text outline (black) then fill (fg)
    cv2.putText(img, text, (x, y), font, font_scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, fg,        2, cv2.LINE_AA)


# ---------- helpers ----------
def ocr_tokens(image, psm):
    """Run Tesseract and return list of digit tokens with conf and bbox."""
    cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
    d = pytesseract.image_to_data(image, config=cfg, output_type=pytesseract.Output.DICT)
    out = []
    n = len(d["text"])
    for i in range(n):
        txt = d["text"][i].strip()
        if not txt or not txt.isdigit(): continue
        conf = float(d["conf"][i]) if d["conf"][i] != "-1" else 0.0
        x,y,w,h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        out.append({"txt":txt, "conf":conf, "bbox":(x,y,w,h)})
    return out

def stitch_line(tokens):
    """Stitch tokens on same line into numbers (joins '5'+'0'->'50')."""
    if not tokens: return []
    toks = sorted(tokens, key=lambda t:(t["bbox"][1], t["bbox"][0]))
    used = [False]*len(toks); lines = []
    for i,a in enumerate(toks):
        if used[i]: continue
        ax,ay,aw,ah = a["bbox"]; ayc = ay+ah/2
        line = [i]; used[i]=True
        for j,b in enumerate(toks):
            if used[j]: continue
            bx,by,bw,bh = b["bbox"]; byc = by+bh/2
            if abs(byc-ayc) <= 0.6*max(ah,bh):  # same row
                line.append(j); used[j]=True
        line = sorted(line, key=lambda k:toks[k]["bbox"][0])
        text = "".join(toks[k]["txt"] for k in line)
        conf = float(np.mean([toks[k]["conf"] for k in line]))
        x1 = min(toks[k]["bbox"][0] for k in line)
        y1 = min(toks[k]["bbox"][1] for k in line)
        x2 = max(toks[k]["bbox"][0]+toks[k]["bbox"][2] for k in line)
        y2 = max(toks[k]["bbox"][1]+toks[k]["bbox"][3] for k in line)
        lines.append({"txt":text, "conf":conf, "bbox":(x1,y1,x2-x1,y2-y1)})
    return lines

def pick_speed(cands):
    """Pick a plausible 10..130 value; prefer 3 digits, higher conf, taller bbox."""
    best = None
    for c in cands:
        text = c["txt"]
        # pull 3 or 2 digit substrings
        nums = re.findall(r"\d{3}|\d{2}", text)
        nums = [int(n) for n in nums if 10 <= int(n) <= 130]
        if not nums: continue
        n = max(nums)
        x,y,w,h = c["bbox"]
        score = (3 if n>=100 else len(str(n)))*100 + c["conf"] + h*0.02
        if best is None or score > best[0]:
            best = (score, n, (x,y,w,h))
    return best  # None or (score, n, bbox)

def lower_band_roi(bw):
    """Fallback ROI: auto-detect lower band by horizontal ink profile."""
    H,W = bw.shape
    proj = (bw>0).sum(axis=1)/W
    proj = cv2.GaussianBlur(proj.reshape(-1,1),(1,9),0).ravel()
    start = int(0.45*H)
    peak  = start + int(np.argmax(proj[start:]))
    y1 = max(int(0.50*H), int(peak-0.12*H))
    y2 = min(H, int(y1+0.35*H))
    return 0,y1,W,y2

# ---------- load & preprocess (GLOBAL) ----------
img  = cv2.imread(IMG)
if img is None:
    raise FileNotFoundError(f"Cannot read {IMG}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# upscale for OCR
scale = 3.0
big  = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

# contrast + two binarizations
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
eq    = clahe.apply(big)
bw_ad = cv2.adaptiveThreshold(eq,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV,31,10)
_, bw_otsu = cv2.threshold(eq,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# thicken slightly (helps Tesseract)
kernel3 = np.ones((3,3),np.uint8)
bw_ad   = cv2.morphologyEx(bw_ad,  cv2.MORPH_CLOSE, kernel3)
bw_otsu = cv2.morphologyEx(bw_otsu,cv2.MORPH_CLOSE, kernel3)

# ---------- multi-pass OCR over whole image ----------
variants = [
    ("bw_ad_psm6",  bw_ad, 6),
    ("bw_ad_psm7",  bw_ad, 7),
    ("bw_otsu_psm6",bw_otsu,6),
    ("gray_psm6",   big,   6),
    ("gray_psm7",   big,   7),
    ("gray_psm11",  big,  11),  # sparse
]

stitched = []
for name, im, psm in variants:
    toks = ocr_tokens(im, psm)
    stitched += stitch_line(toks)

best = pick_speed(stitched)

# ---------- fallback: OCR only the lower band if global failed ----------
if best is None:
    H,W = bw_ad.shape
    x1,y1,x2,y2 = lower_band_roi(bw_ad)
    roi_bw  = bw_ad[y1:y2, x1:x2]
    roi_g   = big[y1:y2, x1:x2]
    # extra cleanup
    roi_bw = cv2.morphologyEx(roi_bw, cv2.MORPH_OPEN, kernel3)
    roi_bw = cv2.morphologyEx(roi_bw, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    # OCR on ROI
    toks = ocr_tokens(roi_bw, 7) + ocr_tokens(roi_g, 7)
    stitched = stitch_line(toks)
    b2 = pick_speed(stitched)
    if b2 is not None:
        # map ROI bbox to global coords
        sc, n, (x,y,w,h) = b2
        best = (sc, n, (x1+x, y1+y, w, h))

# ---- Draw & save with adaptive colors ----
vis = img.copy()
label = "MAX ?"

if best:
    _, speed, (x,y,w,h) = best
    # downscale bbox back to original image (inverse of scale)
    sx, sy, sw, sh = int(x/scale), int(y/scale), int(w/scale), int(h/scale)

    # choose colors based on background under the digits
    mean_bgr = mean_bgr_under(vis, (sx, sy, sw, sh))
    fg = best_contrast_color(mean_bgr)             # text color
    # pick an accent that contrasts with fg (cyan vs yellow)
    accent = (255, 255, 0) if fg == (0,0,0) else (0, 255, 255)  # BGR

    # double-stroke rectangle: outer black (thick), inner accent (thin)
    cv2.rectangle(vis, (sx,sy), (sx+sw,sy+sh), (0,0,0), 6, cv2.LINE_AA)
    cv2.rectangle(vis, (sx,sy), (sx+sw,sy+sh), accent,   2, cv2.LINE_AA)

    label = f"MAX {speed}"
    # text background uses the opposite of fg for maximum contrast
    bg = (255,255,255) if fg == (0,0,0) else (0,0,0)
    draw_text_with_bg(vis, label, (sx, max(30, sy-10)), font_scale=1.1, fg=fg, bg=bg, alpha=0.8)
else:
    # fallback label at top-left with high-contrast styling
    draw_text_with_bg(vis, label, (20,40), font_scale=1.1, fg=(255,255,255), bg=(0,0,0), alpha=0.8)

cv2.imwrite(str(OUT), vis)
print("Saved:", OUT, "| Result:", label)
