"""
readMakoPlannedPlan.py
----------------------
Reader for the Mako *Planned* summary screenshot (testPlanned.jpg).

This is a separate screen from the implant-planning screenshot handled by
readMakoImplantPlan.py, so it gets its own ROI table and parser. It follows
the same overall pattern:

    1. ROIs live in ONE place (_get_planned_roi_params) as fractional x,y,w,h.
    2. Each field is cropped, OCR'd with preprocessing chosen for its text,
       and parsed into a value.
    3. visualize_planned_rois() draws the boxes + raw OCR for tuning.

Fields read (one value each):
    component_size_femur      -> int    (Component Size: Femur)
    component_size_baseplate  -> int    (Component Size: Baseplate)
    component_size_insert     -> float  (Component Size: Insert, in mm)
    planned_alignment         -> float  (degrees; +varus, -valgus)

Usage
-----
    import numpy as np
    from PIL import Image
    from readMakoPlannedPlan import read_mako_planned_plan, visualize_planned_rois

    img = np.array(Image.open('testPlanned.jpg'))
    h, w = img.shape[:2]
    img_4x = np.array(Image.fromarray(img).resize((w*4, h*4), Image.LANCZOS))

    planned = read_mako_planned_plan(img_4x)
    vis     = visualize_planned_rois(img_4x, save_path='planned_roi_annotated.jpg')
"""

import re

import numpy as np
import pytesseract
from PIL import Image, ImageOps

# Reuse shared helpers / Tesseract configuration from the implant-plan reader.
from readMakoImplantPlan import _crop_roi, _otsu_threshold, _safe_float

# ---------------------------------------------------------------------------
# ROI parameters
# ---------------------------------------------------------------------------
#
# Boxes were measured by hand on the 4x-upscaled image (7680 x 4320 px) and
# converted to fractions of image size so they generalize across resolutions:
#
#     x_frac = px_x / 7680      w_frac = px_w / 7680
#     y_frac = px_y / 4320      h_frac = px_h / 4320
#
# Source pixel boxes (top-left -> bottom-right) on the 4x image:
#     femur     : (1610, 1974) -> (2020, 2134)   w=410 h=160
#     baseplate : (1610, 2134) -> (2020, 2294)   w=410 h=160
#     insert    : (1610, 2294) -> (2020, 2454)   w=410 h=160
#     alignment : (1800, 2550) -> (2400, 2700)   w=600 h=150
#     laxity_extension_c1 : (1750, 2890) -> (2150, 3050)   w=400 h=160
#     laxity_flexion_c1 : (1750, 3050) -> (2150, 3110)   w=400 h=60
#     laxity_extension_c2 : (2150, 2980) -> (2550, 3140)   w=400 h=160
#     laxity_flexion_c2 : (2150, 3140) -> (2550, 3300)   w=400 h=160
#     postslope_medial : (5150, 2930) -> (5720, 3090)   w=570 h=160
#     postslope_lateral : (5150, 3090) -> (5720, 3250)   w=570 h=160
#     femoral_dist_resect_c1 : (3320, 2172) -> (3750, 2332)   w=430 h=160
#     femoral_dist_resect_c2 : (3750, 2172) -> (4180, 2332)   w=430 h=160
#     femoral_post_resect_c1 : (3320, 2332) -> (3750, 2492)   w=430 h=160
#     femoral_post_resect_c2 : (3750, 2332) -> (4180, 2492)   w=430 h=160

_REF_W = 7680.0
_REF_H = 4320.0

def _px_box(x0: int, y0: int, x1: int, y1: int) -> tuple:
    """Convert a pixel box on the 4x reference image to fractional (x, y, w, h)."""
    return (
        x0 / _REF_W,
        y0 / _REF_H,
        (x1 - x0) / _REF_W,
        (y1 - y0) / _REF_H,
    )


def _get_planned_roi_params() -> dict:
    """Single source of truth for the planned-screen ROIs (fractional x,y,w,h)."""
    return {
        "component_size_femur":     _px_box(1610, 1974, 2020, 2134),
        "component_size_baseplate": _px_box(1610, 2134, 2020, 2294),
        "component_size_insert":    _px_box(1610, 2294, 2020, 2454),
        "planned_alignment":        _px_box(1800, 2550, 2400, 2700),
        "planned_laxity_extension_c1": _px_box(1750, 2980, 2150, 3140),
        "planned_laxity_extension_c2": _px_box(2150, 2980, 2550, 3140),
        "planned_laxity_flexion_c1": _px_box(1750, 3140, 2150, 3300),
        "planned_laxity_flexion_c2": _px_box(2150, 3140, 2550, 3300),
        "femoral_rotation_coronal": _px_box(3450, 1430, 4050, 1590),
        "femoral_rotation_transverse": _px_box(3450, 1590, 4050, 1750),
        "femoral_rotation_sagittal": _px_box(3450, 1750, 4050, 1910),
        "postslope_medial": _px_box(5150, 2930, 5720, 3090),
        "postslope_lateral": _px_box(5150, 3090, 5720, 3250),
        "femoral_dist_resect_c1" : _px_box(3320, 2172, 3750, 2332),
        "femoral_dist_resect_c2" : _px_box(3750, 2172, 4180, 2332),
        "femoral_post_resect_c1" : _px_box(3320, 2332, 3750, 2492),
        "femoral_post_resect_c2" : _px_box(3750, 2332, 4180, 2492),
        "tibial_rotation_coronal": _px_box(3450, 2770, 4050, 2930),
        "tibial_rotation_transverse": _px_box(3450, 2930, 4050, 3090),
        "tibial_rotation_sagittal": _px_box(3450, 3090, 4050, 3250),
        "tibial_resect_c1" : _px_box(3320, 3510, 3750, 3670),
        "tibial_resect_c2" : _px_box(3750, 3510, 4180, 3670),
    }



# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _prep_crop(crop: Image.Image, scale_to: int = 200) -> Image.Image:
    """
    Produce a clean black-text-on-white image from a white-on-dark ROI.

    The Mako "Plan & Anatomy" screen renders values as light text on a dark
    panel, so we threshold and map the light pixels to black for Tesseract.
    """
    gray = crop.convert("L")

    # Upscale small crops so Tesseract has enough resolution.
    if gray.width < scale_to:
        factor = max(2, (scale_to + gray.width - 1) // max(gray.width, 1))
        gray = gray.resize(
            (gray.width * factor, gray.height * factor), Image.LANCZOS
        )

    gray = ImageOps.autocontrast(gray)
    arr = np.array(gray)
    t = _otsu_threshold(arr)

    # White-on-dark: light pixels (>= threshold) are the text -> render black.
    out = np.where(arr >= t, 0, 255)

    return Image.fromarray(out.astype(np.uint8))


def _ocr_number(crop: Image.Image) -> str:
    """OCR a numeric field (digits + optional decimal). Tries a few PSMs."""
    binary = _prep_crop(crop)
    cfg = "-c tessedit_char_whitelist=0123456789."
    for psm in (7, 8, 6, 13):
        text = pytesseract.image_to_string(binary, config=f"--psm {psm} {cfg}").strip()
        cleaned = re.sub(r"[^\d.]", "", text)
        if re.search(r"\d", cleaned):
            return cleaned
    return ""


def _ocr_alignment(crop: Image.Image) -> str:
    """OCR the alignment field: '<number>° <varus|valgus>' (letters allowed)."""
    binary = _prep_crop(crop, scale_to=320)
    best = ""
    for psm in (7, 6, 11):
        text = pytesseract.image_to_string(binary, config=f"--psm {psm}").strip()
        if len(text) > len(best):
            best = text
    return best


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_int(text: str) -> float:
    """Parse a component size as an integer (NaN on failure)."""
    m = re.search(r"\d+", text)
    return float(int(m.group())) if m else float("nan")


def _parse_alignment(text: str) -> float:
    """
    Parse 'X° varus' / 'X° valgus' into a signed float.

    Convention: varus -> positive, valgus -> negative.
    OCR of 'valgus'/'varus' is fragile, so we match on the first few letters
    after the degree symbol ('val' vs 'var').
    """
    
    if not text:
        return float("nan")

    m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", "."))
    if not m:
        return float("nan")
    value = float(m.group(1))
    low = text.lower()

    if "valg" in low or re.search(r"val", low):
        return -value
    if "varu" in low or re.search(r"var", low):
        return value

    # No readable side keyword — return unsigned magnitude and let caller flag it.
    return value

def _parse_femoral_rotation(text: str) -> float:
    """
    Parse 'X° <varus/valgus|internal/external|flexion/extension>' into a signed float.

    Convention: varus -> positive, internal -> positive, flexion -> positive.
    OCR of 'varus'/'valgus'/'internal'/'external'/'flexion'/'extension' is fragile, so we match on the first few letters
    after the degree symbol ('varus/valgus' vs 'internal/external' vs 'flexion/extension').
    """
    if not text:
        return float("nan")

    m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", "."))
    if not m:
        return float("nan")
    value = float(m.group(1))

    low = text.lower()
    if "valg" in low or re.search(r"valg", low):
        return -value
    if "varu" in low or re.search(r"varu", low):
        return value
    if "intern" in low or re.search(r"intern", low):
        return value
    if "extern" in low or re.search(r"extern", low):
        return -value
    if "flex" in low or re.search(r"flex", low):
        return value
    if "tens" in low or re.search(r"tens", low):
        return -value
    return float("nan")

def _parse_tibial_rotation(text: str) -> float:
    """
    Parse 'X° <varus/valgus|internal/external|p.slope>' into a signed float.

    Convention: varus -> positive, internal -> positive, p.slope -> positive.
    OCR of 'varus'/'valgus'/'internal'/'external'/'p.slope' is fragile, so we match on the first few letters
    after the degree symbol ('varus/valgus' vs 'internal/external' vs 'p.slope').
    """
    if not text:
        return float("nan")

    m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", "."))
    if not m:
        return float("nan")
    value = float(m.group(1))

    low = text.lower()
    if "valg" in low or re.search(r"valg", low):
        return -value
    if "varu" in low or re.search(r"varu", low):
        return value
    if "intern" in low or re.search(r"intern", low):
        return value
    if "extern" in low or re.search(r"extern", low):
        return -value
    if "slope" in low or re.search(r"slope", low):
        return value
    return float("nan")

# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------

def read_mako_planned_plan(img_color: np.ndarray, isright: int = 1) -> dict:
    """
    Read the planned-summary fields from a Mako screenshot.

    Parameters
    ----------
    img_color : np.ndarray (H, W, 3) uint8
        Color image, typically the 4x-upscaled testPlanned.jpg (same scale the
        ROI pixels were measured on).

    Returns
    -------
    dict
        {
            'component_size_femur'     : int-as-float or NaN,
            'component_size_baseplate' : int-as-float or NaN,
            'component_size_insert'    : float (mm) or NaN,
            'planned_alignment'        : float (deg, +varus / -valgus) or NaN,
            'planned_laxity_extension_c1' : float (mm) or NaN,  c1 is lateral if right knee, medial if left knee
            'planned_laxity_flexion_c1' : float (mm) or NaN,    c1 is lateral if right knee, medial if left knee
            'planned_laxity_extension_c2' : float (mm) or NaN,    c2 is lateral if right knee, medial if left knee
            'planned_laxity_flexion_c2' : float (mm) or NaN,    c2 is lateral if right knee, medial if left knee
            'postslope_medial' : float (deg, p.slope) or NaN,
            'postslope_lateral' : float (deg, p.slope) or NaN,
        }
    """
    p = _get_planned_roi_params()
    img_pil = Image.fromarray(img_color).convert("RGB")

    result = {}

    result["component_size_femur"] = _parse_int(
        _ocr_number(_crop_roi(img_pil, p["component_size_femur"]))
    )
    result["component_size_baseplate"] = _parse_int(
        _ocr_number(_crop_roi(img_pil, p["component_size_baseplate"]))
    )
    result["component_size_insert"] = _safe_float(
        _ocr_number(_crop_roi(img_pil, p["component_size_insert"]))
    )
    result["planned_alignment"] = _parse_alignment(
        _ocr_alignment(_crop_roi(img_pil, p["planned_alignment"]))
    )
    result["planned_laxity_extension_c1"] = _safe_float(
        _ocr_number(_crop_roi(img_pil, p["planned_laxity_extension_c1"]))
    )
    result["planned_laxity_flexion_c1"] = _safe_float(
        _ocr_number(_crop_roi(img_pil, p["planned_laxity_flexion_c1"]))
    )
    result["planned_laxity_extension_c2"] = _safe_float(
        _ocr_number(_crop_roi(img_pil, p["planned_laxity_extension_c2"]))
    )
    result["planned_laxity_flexion_c2"] = _safe_float(
        _ocr_number(_crop_roi(img_pil, p["planned_laxity_flexion_c2"]))
    )
    result["femoral_rotation_coronal"] = _parse_femoral_rotation(
        _ocr_alignment(_crop_roi(img_pil, p["femoral_rotation_coronal"]))
    )
    result["femoral_rotation_transverse"] = _parse_femoral_rotation(
        _ocr_alignment(_crop_roi(img_pil, p["femoral_rotation_transverse"]))
    )
    result["femoral_rotation_sagittal"] = _parse_femoral_rotation(
        _ocr_alignment(_crop_roi(img_pil, p["femoral_rotation_sagittal"]))
    )
    result["postslope_medial"] = _parse_alignment(
        _ocr_alignment(_crop_roi(img_pil, p["postslope_medial"]))
    )
    result["postslope_lateral"] = _parse_alignment(
        _ocr_alignment(_crop_roi(img_pil, p["postslope_lateral"]))
    )
    result["femoral_dist_resect_c1"] = _safe_float(
        _ocr_number(_crop_roi(img_pil, p["femoral_dist_resect_c1"]))
    )
    result["femoral_dist_resect_c2"] = _safe_float(
        _ocr_number(_crop_roi(img_pil, p["femoral_dist_resect_c2"]))
    )
    result["femoral_post_resect_c1"] = _safe_float(
        _ocr_number(_crop_roi(img_pil, p["femoral_post_resect_c1"]))
    )
    result["femoral_post_resect_c2"] = _safe_float(
        _ocr_number(_crop_roi(img_pil, p["femoral_post_resect_c2"]))
    )
    result["tibial_rotation_coronal"] = _parse_tibial_rotation(
        _ocr_alignment(_crop_roi(img_pil, p["tibial_rotation_coronal"]))
    )
    result["tibial_rotation_transverse"] = _parse_tibial_rotation(
        _ocr_alignment(_crop_roi(img_pil, p["tibial_rotation_transverse"]))
    )
    result["tibial_rotation_sagittal"] = _parse_tibial_rotation(
        _ocr_alignment(_crop_roi(img_pil, p["tibial_rotation_sagittal"]))
    )
    result["tibial_resect_c1"] = _safe_float(
        _ocr_number(_crop_roi(img_pil, p["tibial_resect_c1"]))
    )
    result["tibial_resect_c2"] = _safe_float(
        _ocr_number(_crop_roi(img_pil, p["tibial_resect_c2"]))
    )

    if isright == 1:
        result["femoral_post_resect_lateral"] = result["femoral_post_resect_c1"]
        result["femoral_post_resect_medial"] = result["femoral_post_resect_c2"]
        result["femoral_dist_resect_lateral"] = result["femoral_dist_resect_c1"]
        result["femoral_dist_resect_medial"] = result["femoral_dist_resect_c2"]
        result["tibial_resect_lateral"] = result["tibial_resect_c1"]
        result["tibial_resect_medial"] = result["tibial_resect_c2"]
        result["planned_laxity_extension_lateral"] = result["planned_laxity_extension_c1"]
        result["planned_laxity_extension_medial"] = result["planned_laxity_extension_c2"]
        result["planned_laxity_flexion_lateral"] = result["planned_laxity_flexion_c1"]
        result["planned_laxity_flexion_medial"] = result["planned_laxity_flexion_c2"]
    else:
        result["femoral_post_resect_lateral"] = result["femoral_post_resect_c2"]
        result["femoral_post_resect_medial"] = result["femoral_post_resect_c1"]
        result["femoral_dist_resect_lateral"] = result["femoral_dist_resect_c2"]
        result["femoral_dist_resect_medial"] = result["femoral_dist_resect_c1"]
        result["tibial_resect_lateral"] = result["tibial_resect_c2"]
        result["tibial_resect_medial"] = result["tibial_resect_c1"]
        result["planned_laxity_extension_lateral"] = result["planned_laxity_extension_c2"]
        result["planned_laxity_extension_medial"] = result["planned_laxity_extension_c1"]
        result["planned_laxity_flexion_lateral"] = result["planned_laxity_flexion_c2"]
        result["planned_laxity_flexion_medial"] = result["planned_laxity_flexion_c1"]
    return result


# ---------------------------------------------------------------------------
# Visualization (tuning aid)
# ---------------------------------------------------------------------------

def visualize_planned_rois(img_color: np.ndarray, save_path: str = None) -> Image.Image:
    """
    Draw each ROI box on the color image with its label and raw OCR text.

    Use this to confirm the boxes land on the right text before trusting the
    parsed values.
    """
    from PIL import ImageDraw, ImageFont

    p = _get_planned_roi_params()
    img_pil = Image.fromarray(img_color).convert("RGB")

    img_draw = img_pil.copy()
    draw = ImageDraw.Draw(img_draw)
    W, H = img_draw.size

    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, H // 70))
    except Exception:
        font = ImageFont.load_default()

    lw = max(3, H // 300)

    # raw OCR text per box for display
    raw_text = {
        "component_size_femur":     _ocr_number(_crop_roi(img_pil, p["component_size_femur"])),
        "component_size_baseplate": _ocr_number(_crop_roi(img_pil, p["component_size_baseplate"])),
        "component_size_insert":    _ocr_number(_crop_roi(img_pil, p["component_size_insert"])),
        "planned_alignment":        _ocr_alignment(_crop_roi(img_pil, p["planned_alignment"])),
        "planned_laxity_extension_c1": _ocr_number(_crop_roi(img_pil, p["planned_laxity_extension_c1"])),
        "planned_laxity_flexion_c1": _ocr_number(_crop_roi(img_pil, p["planned_laxity_flexion_c1"])),
        "planned_laxity_extension_c2": _ocr_number(_crop_roi(img_pil, p["planned_laxity_extension_c2"])),
        "planned_laxity_flexion_c2": _ocr_number(_crop_roi(img_pil, p["planned_laxity_flexion_c2"])),
        "femoral_rotation_coronal": _ocr_alignment(_crop_roi(img_pil, p["femoral_rotation_coronal"])),
        "femoral_rotation_transverse": _ocr_alignment(_crop_roi(img_pil, p["femoral_rotation_transverse"])),
        "femoral_rotation_sagittal": _ocr_alignment(_crop_roi(img_pil, p["femoral_rotation_sagittal"])),
        "postslope_medial": _ocr_alignment(_crop_roi(img_pil, p["postslope_medial"])),
        "postslope_lateral": _ocr_alignment(_crop_roi(img_pil, p["postslope_lateral"])),
        "femoral_dist_resect_c1": _ocr_number(_crop_roi(img_pil, p["femoral_dist_resect_c1"])),
        "femoral_dist_resect_c2": _ocr_number(_crop_roi(img_pil, p["femoral_dist_resect_c2"])),
        "femoral_post_resect_c1": _ocr_number(_crop_roi(img_pil, p["femoral_post_resect_c1"])),
        "femoral_post_resect_c2": _ocr_number(_crop_roi(img_pil, p["femoral_post_resect_c2"])),
        "tibial_rotation_coronal": _ocr_alignment(_crop_roi(img_pil, p["tibial_rotation_coronal"])),
        "tibial_rotation_transverse": _ocr_alignment(_crop_roi(img_pil, p["tibial_rotation_transverse"])),
        "tibial_rotation_sagittal": _ocr_alignment(_crop_roi(img_pil, p["tibial_rotation_sagittal"])),
        "tibial_resect_c1": _ocr_number(_crop_roi(img_pil, p["tibial_resect_c1"])),
        "tibial_resect_c2": _ocr_number(_crop_roi(img_pil, p["tibial_resect_c2"])),
    }

    colors = {
        "component_size_femur":     "#FF4444",
        "component_size_baseplate": "#FFAA00",
        "component_size_insert":    "#44AAFF",
        "planned_alignment":        "#44DD44",
        "planned_laxity_extension_c1": "#44AAFF",
        "planned_laxity_flexion_c1": "#44AAFF",
        "planned_laxity_extension_c2": "#44AAFF",
        "planned_laxity_flexion_c2": "#44AAFF",
        "femoral_rotation_coronal": "#88BBFF",
        "femoral_rotation_transverse": "#88BBFF",
        "femoral_rotation_sagittal": "#88BBFF",
        "postslope_medial": "#44DD44",
        "postslope_lateral": "#44DD44",
        "femoral_dist_resect_c1": "#44AAFF",
        "femoral_dist_resect_c2": "#44AAFF",
        "femoral_post_resect_c1": "#44AAFF",
        "femoral_post_resect_c2": "#44AAFF",
        "tibial_rotation_coronal": "#88BBFF",
        "tibial_rotation_transverse": "#88BBFF",
        "tibial_rotation_sagittal": "#88BBFF",
        "tibial_resect_c1": "#44AAFF",
        "tibial_resect_c2": "#44AAFF",
    }

    for name, roi in p.items():
        x, y, w, h = roi
        x0, y0 = int(x * W), int(y * H)
        x1, y1 = int((x + w) * W), int((y + h) * H)
        color = colors.get(name, "#FFFFFF")
        draw.rectangle([x0, y0, x1, y1], outline=color, width=lw)
        draw.text((x0 + 4, y0 - max(18, H // 60)), name, fill=color, font=font)
        draw.text((x0 + 4, y1 + 4), raw_text.get(name, "") or "??", fill=color, font=font)

    if save_path:
        img_draw.save(save_path)
        print(f"Saved annotated image to: {save_path}")

    return img_draw
