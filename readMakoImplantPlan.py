"""
readMakoImplantPlan.py
----------------------
Python translation of readMakoImplantPlan_v2.m (type == 1 only).

Usage
-----
    import numpy as np
    from PIL import Image
    from readMakoImplantPlan import read_mako_implant_plan, visualize_rois

    img = np.array(Image.open('001_Implant_Planning16.jpg'))
    h, w = img.shape[:2]
    img_4x = np.array(Image.fromarray(img).resize((w*4, h*4), Image.LANCZOS))
    mask = (img_4x[:,:,0] <= 100) & (img_4x[:,:,1] <= 100) & (img_4x[:,:,2] <= 100)

    plan = read_mako_implant_plan(mask, new_version=False)
    vis  = visualize_rois(img_4x, new_version=False, save_path='roi_annotated.jpg')
"""

import re
import numpy as np
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r'C:\Users\pourmodhejir\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
)

# ---------------------------------------------------------------------------
# ROI parameter builder  ← single source of truth used by BOTH functions
# ---------------------------------------------------------------------------

def _get_roi_params(new_version: bool = False) -> dict:
    """
    Return all bounding-box parameters as fractional [x, y, w, h] values
    relative to image size.  This is the SINGLE definition used by both
    read_mako_implant_plan() and visualize_rois().

    Coordinate convention (matches MATLAB):
        x  → column fraction  (left edge)
        y  → row fraction     (top edge)
        w  → width fraction
        h  → height fraction
    """
    p = dict(
        # --- Rotation boxes (one per component × plane) ---
        # Vertical anchor (top edge) for femur / tibia
        rot_v = np.array([0.0722, 0.8750]),
        # Horizontal anchor (left edge) for coronal / axial / sagittal
        rot_h = np.array([0.079,  0.3250, 0.5750]),
        rot_sz = np.array([0.05,  0.0840]),   # [w, h]

        # --- Cut-thickness boxes ---
        # Exact pixel coordinates measured by user on 4x image (7674x4304),
        # converted to fractions.
        cut_v  = np.array([0.4540, 0.5376]),           # femur, tibia (top edge)
        cut_h  = np.array([0.0194, 0.1071, 0.2696, 0.3589]),
        cut_sz = np.array([0.0716, 0.0458]),            # [w, h]

        # --- Implant size boxes ---
        impl_h = 0.9275,
        impl_v = np.array([0.2550, 0.3109]),   # femur, tibia
        impl_sz = np.array([0.0151, 0.0306]),  # [w, h]
    )

    if new_version:
        p['rot_v']  = p['rot_v']  + np.array([0.010, 0.012])
        p['rot_h']  = p['rot_h']  + 0.012
        p['rot_sz'][0] += 0.005
        p['cut_v']  = p['cut_v']  + 0.010
        p['cut_h']  = p['cut_h']  + np.array([0.020, 0.040, 0.020, 0.040])
        p['cut_sz'][0] += 0.020
        p['impl_h'] = p['impl_h'] + 0.021
        p['impl_v'] = p['impl_v'] - 0.070

    return p


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _crop_roi(img_pil: Image.Image, rel_roi: tuple) -> Image.Image:
    """Crop a PIL image using fractional [x, y, w, h] coordinates."""
    x, y, w, h = rel_roi
    W, H = img_pil.size
    return img_pil.crop((int(x*W), int(y*H), int((x+w)*W), int((y+h)*H)))


def _ocr_block(crop: Image.Image) -> list:
    """PSM 6 — uniform block → list of non-empty lines."""
    raw = pytesseract.image_to_string(crop, config='--psm 6')
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def _otsu_threshold(arr: np.ndarray) -> int:
    """Otsu threshold for uint8 grayscale array."""
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, max_var, threshold = 0.0, 0.0, 0.0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var, threshold = var_between, t
    return int(threshold)


def _binarize_cut(img: Image.Image) -> Image.Image:
    """Black digits on white background (Tesseract-friendly)."""
    arr = np.array(img.convert('L'))
    t = _otsu_threshold(arr)
    out = np.where(arr < t, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _normalize_cut_ocr(raw: str) -> str:
    """
    Clean cut-thickness OCR. Fixes common cases:
      '75' -> '7.5' when the decimal was lost
      '4.' -> '4.0', stray 'mm' removed
    """
    if not raw:
        return ''
    s = raw.replace(',', '.').replace('O', '0').replace('o', '0')
    s = re.sub(r'[^\d.]', '', s)
    if not s or s.startswith('.'):
        return ''

    if s.count('.') > 1:
        first, rest = s.split('.', 1)
        s = first + '.' + rest.replace('.', '')

    if s.endswith('.'):
        s = s + '0'

    # Lost decimal: Tesseract often reads 7.5 as 75 (or 8.0 as 80)
    if '.' not in s and re.fullmatch(r'\d{2}', s):
        v = int(s)
        if 40 <= v <= 99:
            v10 = v / 10.0
            if 3.0 <= v10 <= 15.0:
                return f'{v10:.1f}'

    return s


def _pick_best_cut_ocr(candidates: list) -> str:
    """Choose the most plausible mm value from several OCR attempts."""
    # Allow smaller valid cuts like 2.5 while still rejecting tiny OCR noise.
    min_mm = 2.0
    best_text, best_score = '', -1.0
    for raw in candidates:
        text = _normalize_cut_ocr(raw)
        if not text:
            continue
        try:
            v = float(text)
        except ValueError:
            continue
        if not (min_mm <= v <= 15.0):
            continue
        score = 10.0
        if '.' in text:
            score += 3.0
        if re.match(r'^\d+\.\d$', text):
            score += 4.0
        if len(text) >= 3:
            score += 1.0
        if score > best_score:
            best_score, best_text = score, text
    return best_text


def _ocr_line(crop: Image.Image) -> str:
    """
    OCR a cut label like '7.5mm' from a color ROI.

    These labels are white text on a dark panel (not the small white boxes
  used elsewhere). Use the full ROI, invert to black-on-white, then PSM 8.
    """
    from PIL import ImageOps

    gray = crop.convert('L')
    scale = max(4, (200 + gray.width - 1) // max(gray.width, 1))
    big = gray.resize((gray.width * scale, gray.height * scale), Image.LANCZOS)
    big = ImageOps.autocontrast(big)
    big = ImageOps.invert(big)          # white-on-dark → black-on-white
    binary = _binarize_cut(big)

    cfg = '-c tessedit_char_whitelist=0123456789.'
    # PSM 8 (single word) reads 7.5 reliably; PSM 7 often returns '.5' / '.9'
    candidates = [
        pytesseract.image_to_string(binary, config=f'--psm {psm} {cfg}').strip()
        for psm in (8, 13, 7)
    ]
    return _pick_best_cut_ocr(candidates)


def _ocr_char(crop: Image.Image) -> str:
    """PSM 10 — single character."""
    return pytesseract.image_to_string(crop, config='--psm 10').strip()


def _strip_degree(s: str) -> str:
    return re.sub(r'[°\u00b0\s]+$', '', s).strip()


def _safe_float(s: str) -> float:
    try:
        # Strip anything that's not a digit or decimal point
        cleaned = re.sub(r'[^\d.]', '', _strip_degree(s))
        return float(cleaned)
    except (ValueError, TypeError):
        return float('nan')


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------

def read_mako_implant_plan(imag: np.ndarray, img_color: np.ndarray,
                           new_version: bool = False) -> np.ndarray:
    """
    Extract implant planning metrics from a masked Mako screenshot.

    Parameters
    ----------
    imag : np.ndarray (H x W) bool or uint8
        Binary mask — True/1 where text pixels are (black text on white bg).
        Used for rotation and implant size OCR.
    img_color : np.ndarray (H x W x 3) uint8
        Original color image at the same scale as imag.
        Used for cut-thickness OCR (white text on dark panel).
    new_version : bool
        False → Mako 2.0,  True → Mako 3.0

    Returns
    -------
    plan : np.ndarray (2, 8)
        Row 0 = femur, row 1 = tibia
        Cols: [coronal_rot, axial_rot, sagittal_rot, cut1, cut2, cut3, cut4, size]
        Sign convention (positive):
            Femur : valgus, external rotation, flexion
            Tibia : varus,  internal rotation, posterior slope
    """
    p = _get_roi_params(new_version)

    img_u8      = (imag.astype(np.uint8) * 255) if imag.dtype == bool else imag.astype(np.uint8)
    img_pil     = Image.fromarray(img_u8).convert('L')       # mask → rotations & size
    img_color_pil = Image.fromarray(img_color).convert('RGB') # color → cut boxes

    plan = np.zeros((2, 8))

    for i in range(2):   # 0 = femur, 1 = tibia

        # Rotations (cols 0-2)
        for j in range(3):
            roi  = (p['rot_h'][j], p['rot_v'][i], p['rot_sz'][0], p['rot_sz'][1])
            crop = _crop_roi(img_pil, roi)
            lines = _ocr_block(crop)
            try:
                label_line = lines[i]
                value_line = lines[1 - i]
            except IndexError:
                plan[i, j] = float('nan')
                continue
            sign = -1 if any(kw in label_line for kw in ('Val', 'Ext')) else 1
            plan[i, j] = sign * _safe_float(value_line)

        # Cuts (cols 3-6) — use color image; _ocr_line auto-crops to white box
        for j in range(4):
            roi  = (p['cut_h'][j], p['cut_v'][i], p['cut_sz'][0], p['cut_sz'][1])
            crop = _crop_roi(img_color_pil, roi)
            text = _ocr_line(crop)
            plan[i, 3 + j] = _safe_float(text)

        # Implant size (col 7)
        roi  = (p['impl_h'], p['impl_v'][i], p['impl_sz'][0], p['impl_sz'][1])
        crop = _crop_roi(img_pil, roi)
        text = _ocr_char(crop)
        plan[i, 7] = _safe_float(text)

    return plan


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_rois(img_color: np.ndarray, new_version: bool = False,
                   save_path: str = None) -> Image.Image:
    """
    Draw all OCR bounding boxes on the original color image with labels
    and the raw OCR text read from each box.

    Parameters
    ----------
    img_color : np.ndarray (H x W x 3) uint8
        Original color image at the same scale passed to read_mako_implant_plan.
    new_version : bool
        Must match the flag passed to read_mako_implant_plan.
    save_path : str, optional
        If given, saves the annotated image here.

    Returns
    -------
    Annotated PIL Image.
    """
    from PIL import ImageDraw, ImageFont

    p = _get_roi_params(new_version)   # ← same params, no duplication

    # Build mask for OCR
    mask = (
        (img_color[:, :, 0] <= 100) &
        (img_color[:, :, 1] <= 100) &
        (img_color[:, :, 2] <= 100)
    )
    img_u8  = (mask.astype(np.uint8)) * 255
    img_ocr = Image.fromarray(img_u8).convert('L')
    img_color_pil = Image.fromarray(img_color).convert('RGB')

    img_draw = Image.fromarray(img_color).convert('RGB')
    draw = ImageDraw.Draw(img_draw)
    W, H = img_draw.size

    try:
        font       = ImageFont.truetype("arial.ttf", size=max(12, H // 80))
        font_small = ImageFont.truetype("arial.ttf", size=max(10, H // 100))
    except Exception:
        font = font_small = ImageFont.load_default()

    ROT_COLOR  = '#FF4444'
    CUT_COLOR  = '#44AAFF'
    IMPL_COLOR = '#44DD44'

    component  = ['Femur', 'Tibia']
    rot_labels = ['Coronal', 'Axial', 'Sagittal']
    cut_labels = ['Cut1', 'Cut2', 'Cut3', 'Cut4']
    lw = max(3, H // 300)

    def draw_box(rel_roi, color, label, ocr_text):
        x, y, w, h = rel_roi
        x0, y0 = int(x * W), int(y * H)
        x1, y1 = int((x + w) * W), int((y + h) * H)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=lw)
        draw.text((x0 + 4, y0 - max(14, H // 70)), label,   fill=color, font=font_small)
        draw.text((x0 + 4, y0 + 4),                ocr_text, fill=color, font=font)

    for i in range(2):
        comp = component[i]

        for j in range(3):
            roi = (p['rot_h'][j], p['rot_v'][i], p['rot_sz'][0], p['rot_sz'][1])
            lines = _ocr_block(_crop_roi(img_ocr, roi))
            draw_box(roi, ROT_COLOR, f'{comp} {rot_labels[j]}', ' | '.join(lines[:3]) or '??')

        for j in range(4):
            roi = (p['cut_h'][j], p['cut_v'][i], p['cut_sz'][0], p['cut_sz'][1])
            draw_box(
                roi, CUT_COLOR, f'{comp} {cut_labels[j]}',
                _ocr_line(_crop_roi(img_color_pil, roi)) or '??',
            )

        roi = (p['impl_h'], p['impl_v'][i], p['impl_sz'][0], p['impl_sz'][1])
        draw_box(roi, IMPL_COLOR, f'{comp} Size', _ocr_char(_crop_roi(img_ocr, roi)) or '??')

    if save_path:
        img_draw.save(save_path)
        print(f"Saved annotated image to: {save_path}")

    return img_draw
