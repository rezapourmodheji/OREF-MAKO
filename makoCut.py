"""
makoCut.py
----------

Builds the Mako and cut-plane coordinate systems from anatomical landmarks
and the surgical plan. Tibia only for now (femur branch included but not yet
exercised).
"""

import numpy as np


def tib_csys(landmarks: dict, plan: dict, side: bool) -> tuple:
    """
    Build coordinate systems from landmarks and surgical plan.

    Parameters
    ----------
    landmarks : dict
        Landmark name → np.array([X, Y, Z]) in CT space.
        Tibia needs: Tibia_Knee_Center, Ankle_Center,
                     PCL_Insertion, Medial_Third_Tubercle,
                     Landmark_Medial
    plan : dict
        Surgical plan values.
        Tibia keys : slope, varus, internal, medialCut
    side : bool
        True = right knee, False = left knee.

    Returns
    -------
    csys : dict
        si, ap, ml  — unit axes in CT space
        Rmako (3x3), Tmako (3x1) — CT → Mako rotation + translation
        Rcut  (3x3), Tcut  (3x1) — Mako → cut-plane rotation + translation
    landmarks_mako : dict
        All landmarks re-expressed in the Mako coordinate system.
    """

    csys = {}

    # ── 1  ANATOMICAL AXES IN CT SPACE ───────────────────────────────────────
    # Superior-inferior: ankle → knee
    si = landmarks['TibKneeCenter'] - landmarks['AnkleCenter']
    csys['si'] = si / np.linalg.norm(si)

    # Project PCL and tubercle onto the axial plane at the knee center
    # Projection formula: p_proj = p - dot(p - origin, si) * si
    kc = landmarks['TibKneeCenter']
    pcl_proj = (landmarks['PCLInsertion'] -
                np.dot(landmarks['PCLInsertion'] - kc, csys['si']) * csys['si'])
    tub_proj = (landmarks['MedThirdTub'] -
                np.dot(landmarks['MedThirdTub'] - kc, csys['si']) * csys['si'])

    # Anterior-posterior: PCL → tubercle (in axial plane)
    ap = tub_proj - pcl_proj
    csys['ap'] = ap / np.linalg.norm(ap)

    # Medial-lateral: right-hand rule
    csys['ml'] = np.cross(csys['ap'], csys['si'])

    xrot = plan['slope']
    yrot = plan['varus']
    zrot = plan['internal']
    tcut = plan['medialcut']

    # ── 2  CT → MAKO TRANSFORM ───────────────────────────────────────────────
    # Rmako rows are the three anatomical axes expressed in CT space.
    # This re-expresses any CT vector in the Mako (ML, AP, SI) frame.
    Rmako = np.vstack([csys['ml'], csys['ap'], csys['si']])   # (3x3)

    # Translation: express the knee center in the Mako frame
    # After this transform, the knee center maps to the origin.
    # Note: uses Tibia_Knee_Center for both tibia and femur branches in MATLAB by Fernando's design
    Tmako = Rmako @ (-landmarks['TibKneeCenter'])         # (3,)

    csys['Rmako'] = Rmako
    csys['Tmako'] = Tmako.reshape(3, 1)   # store as column vector, matches MATLAB

    # ── 3  TRANSFORM ALL LANDMARKS TO MAKO SPACE ────────────────────────────
    landmarks_mako = {}
    T_4x4 = np.eye(4)
    T_4x4[:3, :3] = Rmako
    T_4x4[:3,  3] = Tmako

    for name, coords in landmarks.items():
        p_h = np.append(coords, 1.0)          # homogeneous
        landmarks_mako[name] = (T_4x4 @ p_h)[:3]

    
    # ── 4  CUT-PLANE ROTATION ────────────────────────────────────────────────
    # Sign conventions (right knee positive):
    #   xrot > 0 → posterior slope   (rotates bone around -X)
    #   yrot > 0 → varus             (right: +Y, left: -Y)
    #   zrot > 0 → internal rotation (right: -Z, left: +Z)
    # Rotation order: sagittal (X) first, then coronal (Y), then axial (Z)
    # Each new rotation is PRE-multiplied

    side_sign = (-1) ** (int(side))   # side=True(1) → -1, side=False(0) → +1

    xrot = -xrot                        # posterior slope rotates around -X
    yrot = -yrot * side_sign            # varus: right → +Y, left → -Y
    zrot =  zrot * side_sign            # internal: right → -Z, left → +Z

    def Rx(deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s,  c]])

    def Ry(deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]])

    def Rz(deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]])

    # Rcut = Rz * Ry * Rx  (pre-multiply each new rotation)
    Rcut = Rx(xrot)
    Rcut = Ry(yrot) @ Rcut
    Rcut = Rz(zrot) @ Rcut

    # Translation: place origin at medial landmark, then offset by cut thickness
    Tcut = Rcut @ (-landmarks_mako['MedTibPlateau']) + np.array([0, 0, tcut])

    csys['Rcut'] = Rcut
    csys['Tcut'] = Tcut.reshape(3, 1)

    return csys, landmarks_mako





def fem_csys(landmarks: dict, plan: dict, side: bool) -> tuple:
    """
    Build coordinate systems from landmarks and surgical plan.

    Parameters
    ----------
    landmarks : dict
        Landmark name → np.array([X, Y, Z]) in CT space.
        Femur needs: HipCenter, FemKneeCenter,
                     LatEpicondyle, MedEpicondyle,
                     FemPostMed, FemPostLat
                     FemDistMed, FemDistLat
    plan : dict
        Surgical plan values.
        Femur keys : flexion,  varus, internal, medialCut, posteriorCut
    side : bool
        True = right knee, False = left knee.

    Returns
    -------
    csys : dict
        si, ap, ml  — unit axes in CT space
        Rmako (3x3), Tmako (3x1) — CT → Mako rotation + translation
        Rcut  (3x3), Tcut  (3x1) — Mako → cut-plane rotation + translation
    landmarks_mako : dict
        All landmarks re-expressed in the Mako coordinate system.
    """

    csys = {}

    # ── 1  ANATOMICAL AXES IN CT SPACE ───────────────────────────────────────
    side_sign = (-1) ** (int(side))   # side=True(1) → -1, side=False(0) → +1
    si = landmarks['HipCenter'] - landmarks['FemKneeCenter']
    csys['si'] = si / np.linalg.norm(si)

    LatEpicondyle_proj = (landmarks['LatEpicondyle'] -
                          np.dot(landmarks['LatEpicondyle'] - landmarks['FemKneeCenter'], csys['si']) * csys['si'])
    MedEpicondyle_proj = (landmarks['MedEpicondyle'] -
                          np.dot(landmarks['MedEpicondyle'] - landmarks['FemKneeCenter'], csys['si']) * csys['si'])

    ml = (MedEpicondyle_proj - LatEpicondyle_proj)*side_sign # always to the right
    csys['ml'] = ml / np.linalg.norm(ml)

    csys['ap'] = np.cross(csys['si'], csys['ml'])

    
    
    # ── 2  CT → MAKO TRANSFORM ───────────────────────────────────────────────
    # Rmako rows are the three anatomical axes expressed in CT space.
    # This re-expresses any CT vector in the Mako (ML, AP, SI) frame.
    Rmako = np.vstack([csys['ml'], csys['ap'], csys['si']])   # (3x3)

    # Translation: express the knee center in the Mako frame
    # After this transform, the knee center maps to the origin.
    # Note: uses Tibia_Knee_Center for both tibia and femur branches in MATLAB by Fernando's design
    Tmako = Rmako @ (-landmarks['FemKneeCenter'])         # (3,)

    

    

    csys['Rmako'] = Rmako
    csys['Tmako'] = Tmako.reshape(3, 1) 



    # ── 3  TRANSFORM ALL LANDMARKS TO MAKO SPACE ────────────────────────────
    landmarks_mako = {}
    T_4x4 = np.eye(4)
    T_4x4[:3, :3] = Rmako
    T_4x4[:3,  3] = Tmako

    for name, coords in landmarks.items():
        p_h = np.append(coords, 1.0)          # homogeneous
        landmarks_mako[name] = (T_4x4 @ p_h)[:3]

    
    # ── 4  CUT-PLANE ROTATION ────────────────────────────────────────────────
    # Sign conventions (right knee positive):
    #   xrot > 0 → posterior slope   (rotates bone around -X)
    #   yrot > 0 → varus             (right: +Y, left: -Y)
    #   zrot > 0 → internal rotation (right: -Z, left: +Z)
    # Rotation order: sagittal (X) first, then coronal (Y), then axial (Z)
    # Each new rotation is PRE-multiplied

    xrot = -plan['flexion']
    yrot = -plan['varus']*side_sign
    zrot = -plan['internal']*side_sign

    def Rx(deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s,  c]])

    def Ry(deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]])

    def Rz(deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]])
    
    Rcut = Rx(xrot)
    Rcut = Ry(yrot) @ Rcut
    Rcut = Rz(zrot) @ Rcut

    TcutPostMed = Rcut @ (-landmarks_mako['FemPostMed'] + np.array([0, -plan['medialpostcut'], 0]))
    TcutDistMed = Rcut @ (-landmarks_mako['FemDistMed'] + np.array([0, 0, -plan['medialdistcut']]))

    csys['Rcut'] = Rcut
    csys['TcutPostMed'] = TcutPostMed.reshape(3, 1)
    csys['TcutDistMed'] = TcutDistMed.reshape(3, 1)

    Tpost_4x4 = np.eye(4)
    Tpost_4x4[:3, :3] = Rcut
    Tpost_4x4[:3,  3] = TcutPostMed
    
    Tdist_4x4 = np.eye(4)
    Tdist_4x4[:3, :3] = Rcut
    Tdist_4x4[:3,  3] = TcutDistMed
    

    return csys, landmarks_mako