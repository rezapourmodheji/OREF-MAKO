"""
alignImplantToCut.py
--------------------
Python translation of alignImplantToCut.m

Fits the implant contour to the bone cut profile by minimizing:
  - Overhang (bone points outside implant footprint)
  - Distance to anterior cortex
  - Medial-lateral imbalance
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree


def align_implant_to_cut(implant: np.ndarray,
                         cut_profile: np.ndarray,
                         right: bool) -> np.ndarray:
    """
    Fit implant contour to bone cut profile.

    Parameters
    ----------
    implant : np.ndarray, shape (N, 2) or (N, 3)
        Implant contour points. Only XY columns are used (Z ignored if present).
    cut_profile : np.ndarray, shape (K, 2)
        Ordered boundary points of the tibial cut surface.
    right : bool
        True = right knee, False = left knee.

    Returns
    -------
    trm : np.ndarray, shape (4, 4)
        Homogeneous transform from bone to implant coordinate system.
    """
    implant  = implant[:, :2]      # use only XY — ignore Z column if present
    bone     = cut_profile[:, :2]

    ctr_cut     = bone.mean(axis=0)
    ctr_implant = 0.5 * (implant.max(axis=0) + implant.min(axis=0))

    # Initial shift: align implant centroid to bone centroid
    # Before the optimizer even starts, the implant is shifted so its geometric center sits on the bone centroid.
    # This gives the optimizer a sensible starting point rather than working from an arbitrary position.
    # Note that ctr_implant uses the bounding box center (midpoint of min and max), not the mean — 
    # --- this is intentional because implant contours are not uniformly sampled, so the mean would be biased toward densely sampled regions.
    implant_shifted = implant + (ctr_cut - ctr_implant)

    # ── OBJECTIVE FUNCTION ────────────────────────────────────────────────────
    def area_overlap(x):
        angle, tx, ty = x
        angle = 0.0
        # Build 2D rigid transform (rotation + translation)
        c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        R = np.array([[c, -s, 0, tx],
                      [s,  c, 0, ty],
                      [0,  0, 1,  0],
                      [0,  0, 0,  1]])

        # Transform implant
        impl_h   = np.hstack([implant_shifted,
                              np.zeros((len(implant_shifted), 1)),
                              np.ones((len(implant_shifted), 1))])
        impl_trm = (R @ impl_h.T).T[:, :2]

        # ── Overhang: bone points inside implant, penalised by distance to edge
        from matplotlib.path import Path as MplPath
        impl_path = MplPath(impl_trm)
        inside    = impl_path.contains_points(bone)
        if inside.any():
            bone_in  = bone[inside]
            kdt_impl = KDTree(impl_trm)
            dists, _ = kdt_impl.query(bone_in)
            over     = dists.sum()
        else:
            over = 0.0

        # ── Anterior flush: top 30% of implant should be close to bone edge
        ant_thresh  = np.percentile(impl_trm[:, 1], 70)  # top 30% = Y > 70th percentile
        impl_ant    = impl_trm[impl_trm[:, 1] > ant_thresh]
        kdt_bone    = KDTree(bone)
        _, idx_ant  = kdt_bone.query(impl_ant)
        bone_ant    = bone[idx_ant]
        area_ant    = np.sqrt(((impl_ant - bone_ant + 0.5) ** 2).sum(axis=1)).sum()

        # ── Medial-lateral balance
        # right knee: medial = negative X side; left: medial = positive X side
        sign        = (-1) ** (1 + int(right))
        med_thresh  = np.percentile(impl_trm[:, 0], 15)
        lat_thresh  = np.percentile(impl_trm[:, 0], 85)

        impl_med    = impl_trm[sign * impl_trm[:, 0] < med_thresh]
        impl_lat    = impl_trm[sign * impl_trm[:, 0] > lat_thresh]

        _, idx_med  = kdt_bone.query(impl_med)
        _, idx_lat  = kdt_bone.query(impl_lat)
        bone_med    = bone[idx_med]
        bone_lat    = bone[idx_lat]

        area_med    = np.sqrt(((impl_med - bone_med) ** 2).sum(axis=1)).sum()
        area_lat    = np.sqrt(((impl_lat - bone_lat) ** 2).sum(axis=1)).sum()

        # return 100.0 * over + abs(area_med - area_lat) #100.0 * over + area_ant + abs(area_med - area_lat)
        return 100.0 * over + area_ant + abs(area_med - area_lat)

    # ── CONSTRAINED OPTIMISATION ──────────────────────────────────────────────
    # scipy: minimize with bounds, SLSQP method (handles box constraints)
    x0     = [0.0, 0.0, 0.0]          # [rotation_deg, tx_mm, ty_mm]
    bounds = [(-4, 4), (-10, 10), (-10, 10)]

    result = minimize(area_overlap, x0, method='SLSQP', bounds=bounds,
                      options={'disp': False, 'ftol': 1e-9})
    x = result.x

    # ── BUILD OUTPUT TRANSFORM ────────────────────────────────────────────────
    # MATLAB returns inv(trm) — transform from bone to implant
    # trm(1:2,4) adjusted by -(ctr_cut - ctr_implant)
    c, s = np.cos(np.radians(x[0])), np.sin(np.radians(x[0]))
    trm  = np.array([[c, -s, 0, x[1]],
                     [s,  c, 0, x[2]],
                     [0,  0, 1,  0  ],
                     [0,  0, 0,  1  ]])
    trm        = np.linalg.inv(trm)
    trm[:2, 3] = trm[:2, 3] - (ctr_cut - ctr_implant)
    # trm[:2, 3] = trm[:2, 3] - trm[:2, :2] @ (ctr_cut - ctr_implant)


    
    # trm[:2, 3] = trm[:2, 3] - (ctr_cut - ctr_implant)
    # CORRECT - rotate the correction into the transformed frame
    
    return trm