# -*- coding: utf-8 -*-
"""
processMAKO.py
Reza Pourmodheji - June 2026

Processes MAKO data for OREFTKA modeling projects
"""

from pathlib import Path
import numpy as np
from PIL import Image
from readMakoPlannedPlan import read_mako_planned_plan, visualize_planned_rois
from registrations import tfm2euler123, write_tfm_file
from iovis import read_mako_points
from makoCut import tib_csys, fem_csys
import pyvista as pv
verbose = True
import matplotlib.pyplot as plt
from alignImplantToCut import align_implant_to_cut

# ── FUNCTIONS ──────────────────────────────────────────────────────────────────

LANDMARK_LABELS = {
    'MedMalleolus': 'Med malleolus',
    'LatMalleolus': 'Lat malleolus',
    'PCLInsertion': 'PCL',
    'MedThirdTub': 'Med 1/3 tubercle',
    'TibKneeCenter': 'Tibial Knee center',
    'LatTibPlateau': 'Lat Tibial Plateau',
    'MedTibPlateau': 'Med Tibial Plateau',
    'AnkleCenter': 'Ankle Center',
}


def _add_coord_frame(plotter, origin, axes, arrow_scale, colors, name_prefix):
    """Draw origin (sphere) and X/Y/Z arrows; axes rows are unit vectors in CT space."""
    origin = np.asarray(origin, dtype=float).ravel()
    axes = np.asarray(axes, dtype=float).reshape(3, 3)
    plotter.add_mesh(
        pv.Sphere(radius=arrow_scale * 0.08, center=origin),
        color=colors[0],
        name=f'{name_prefix}_origin',
    )
    for axis, color, label in zip(axes, ('red', 'lime', 'blue'), ('x', 'y', 'z')):
        direction = axis / np.linalg.norm(axis)
        arrow = pv.Arrow(start=origin, direction=direction, scale=arrow_scale)
        plotter.add_mesh(arrow, color=color, name=f'arrow_{name_prefix}_{label}')


def _add_landmarks(plotter, landmarks_to_plot, arrow_scale):
    """
    Plot landmarks as world-space spheres plus text labels (CT space).

    landmarks_to_plot : dict
        Landmark name → (3,) coordinates, e.g. selectlandmarks from check_csys.
    """
    if not landmarks_to_plot:
        return

    # Same mm radius as frame-origin markers (not point_size — that is pixels)
    sphere_radius = arrow_scale * 0.08
    names = list(landmarks_to_plot.keys())
    pts = np.array([landmarks_to_plot[n] for n in names], dtype=float)

    for name, center in zip(names, pts):
        plotter.add_mesh(
            pv.Sphere(radius=sphere_radius, center=center),
            color='magenta',
            name=f'landmark_{name}',
        )

    cloud = pv.PolyData(pts)
    plotter.add_point_labels(
        cloud,
        [LANDMARK_LABELS.get(n, n) for n in names],
        font_size=12,
        show_points=False,
        shape_opacity=0.4,
        always_visible=True,
    )


def check_tibcsys(pfolder, subject, csys, landmarks, arrow_scale=None):
    """
    Plot bone STL in CT space with three coordinate frames overlaid.

    All geometry stays in CT (scanner) coordinates — the same frame as the STL
    and MakoPoints.txt landmarks.

    Frames shown
    ------------
    CT   : origin at (0, 0, 0), axes = scanner X, Y, Z
    Mako : origin at tibia knee center; axes = ML, AP, SI (rows of Rmako)
    Cut  : origin and axes of the cut plane (Mako → cut transform composed
           with CT → Mako), i.e. the frame used for CT_HU / density mapping
    """
    stl_path = pfolder / 'Tib.stl'
    bone_mesh = pv.read(stl_path)

    if arrow_scale is None:
        bounds = np.array(bone_mesh.bounds).reshape(3, 2)
        arrow_scale = 0.15 * np.max(bounds[:, 1] - bounds[:, 0])

    Rmako = csys['Rmako']
    Tmako = csys['Tmako'].ravel()
    Rcut = csys['Rcut']
    Tcut = csys['Tcut'].ravel()
    
    # MAKO
    origin_mako_ct = np.asarray(landmarks['TibKneeCenter'], dtype=float).ravel()
    axes_mako_ct = np.vstack([csys['ml'], csys['ap'], csys['si']])

    origin_cut_cut = np.zeros(3)  # cut plane origin in cut frame is always (0, 0, 0)
    # p_cut = Rcut * p_mako + Tcut
    origin_cut_mako = Rcut.T @ (origin_cut_cut - Tcut)
    # p_mako = Rmako * p_ct + Tmako
    origin_cut_ct = Rmako.T @ (origin_cut_mako - Tmako)
    # CUT
    # Cut basis vectors in CT: rows of (Rcut @ Rmako)
    # origin_cut_mako = -Rcut.T @ Tcut 
    R_ct_to_cut = Rcut @ Rmako
    axes_cut_ct = R_ct_to_cut
    
    

    plotter = pv.Plotter()
    plotter.add_mesh(bone_mesh, color='lightgrey', opacity=0.7, label='bone STL')

    _add_coord_frame(
        plotter, np.zeros(3), np.eye(3), arrow_scale,
        colors=('white',), name_prefix='CT',
    )
    _add_coord_frame(
        plotter, origin_mako_ct, axes_mako_ct, arrow_scale,
        colors=('gold',), name_prefix='Mako',
    )
    _add_coord_frame(
        plotter, origin_cut_ct, axes_cut_ct, arrow_scale,
        colors=('cyan',), name_prefix='Cut',
    )
    selectlandmarks = [
        'PCLInsertion',
        'MedThirdTub',
        'TibKneeCenter',
        'LatTibPlateau',
        'MedTibPlateau',
    ]
    missing = [n for n in selectlandmarks if n not in landmarks]
    if missing:
        print(f'  WARNING: landmarks missing for plot: {", ".join(missing)}')
    landmarks_to_plot = {n: landmarks[n] for n in selectlandmarks if n in landmarks}
    _add_landmarks(plotter, landmarks_to_plot, arrow_scale)

    plotter.add_legend(
        [
            ['CT (scanner)', 'white'],
            ['Mako (knee center)', 'gold'],
            ['Cut plane', 'cyan'],
            ['landmarks', 'magenta'],
            ['bone STL', 'lightgrey'],
        ],
        bcolor='black',
    )
    plotter.add_axes()

    # Superior view: camera 200 mm above knee center (+Z), looking down (−Z)
    cam_height_mm = 200.0
    cam_pos = origin_mako_ct + np.array([0.0, 0.0, cam_height_mm])
    plotter.camera_position = [
        tuple(cam_pos),
        tuple(origin_mako_ct),
        (0.0, 1.0, 0.0),   # screen "up" = +Y (must not be parallel to view axis)
    ]

    plotter.show()
    # plotter.save_graphic(pfolder / f'Tib_csys_check.svg')


def check_femcsys(pfolder, subject, csys, landmarks, arrow_scale=None):
    """
    Plot bone STL in CT space with three coordinate frames overlaid.

    All geometry stays in CT (scanner) coordinates — the same frame as the STL
    and MakoPoints.txt landmarks.

    Frames shown
    ------------
    CT   : origin at (0, 0, 0), axes = scanner X, Y, Z
    Mako : origin at tibia knee center; axes = ML, AP, SI (rows of Rmako)
    Cut  : origin and axes of the cut plane (Mako → cut transform composed
           with CT → Mako), i.e. the frame used for CT_HU / density mapping
    """
    stl_path = pfolder / 'Fem.stl'
    bone_mesh = pv.read(stl_path)

    if arrow_scale is None:
        bounds = np.array(bone_mesh.bounds).reshape(3, 2)
        arrow_scale = 0.15 * np.max(bounds[:, 1] - bounds[:, 0])

    Rmako = csys['Rmako']
    Tmako = csys['Tmako'].ravel()
    Rcut = csys['Rcut']
    TcutPostMed = csys['TcutPostMed'].ravel()
    TcutDistMed = csys['TcutDistMed'].ravel()
    
    # MAKO
    origin_mako_ct = np.asarray(landmarks['FemKneeCenter'], dtype=float).ravel()
    axes_mako_ct = np.vstack([csys['ml'], csys['ap'], csys['si']])
    


    # POST MEDIAL CUT
    origin_cut_cut = np.zeros(3)  # cut plane origin in cut frame is always (0, 0, 0)
    # p_cut = Rcut * p_mako + Tcut
    origin_cut_mako = Rcut.T @ (origin_cut_cut - TcutPostMed)
    # p_mako = Rmako * p_ct + Tmako
    origin_postcut_ct = Rmako.T @ (origin_cut_mako - Tmako)
    # CUT
    # Cut basis vectors in CT: rows of (Rcut @ Rmako)
    # origin_cut_mako = -Rcut.T @ Tcut 
    R_ct_to_cutPostMed = Rcut @ Rmako
    axes_postcut_ct = R_ct_to_cutPostMed
    
    # DISTAL CUT
    origin_cut_cut = np.zeros(3)  # cut plane origin in cut frame is always (0, 0, 0)
    # p_cut = Rcut * p_mako + Tcut
    origin_cut_mako = Rcut.T @ (origin_cut_cut - TcutDistMed)
    # p_mako = Rmako * p_ct + Tmako
    origin_distcut_ct = Rmako.T @ (origin_cut_mako - Tmako)
    # CUT
    # Cut basis vectors in CT: rows of (Rcut @ Rmako)
    # origin_cut_mako = -Rcut.T @ Tcut 
    R_ct_to_cutDistMed = Rcut @ Rmako
    axes_distcut_ct = R_ct_to_cutPostMed
    
    

    plotter = pv.Plotter()
    plotter.add_mesh(bone_mesh, color='lightgrey', opacity=0.7, label='bone STL')

    _add_coord_frame(
        plotter, np.zeros(3), np.eye(3), arrow_scale,
        colors=('white',), name_prefix='CT',
    )
    _add_coord_frame(
        plotter, origin_mako_ct, axes_mako_ct, arrow_scale,
        colors=('gold',), name_prefix='Mako',
    )
    _add_coord_frame(
        plotter, origin_postcut_ct, axes_postcut_ct, arrow_scale,
        colors=('cyan',), name_prefix='Cut Post Medial',
    )
    _add_coord_frame(
        plotter, origin_distcut_ct, axes_distcut_ct, arrow_scale,
        colors=('magenta',), name_prefix='Cut Dist Medial',
    )
    selectlandmarks = [
        'FemPostMed',
        'FemDistMed',
        'FemKneeCenter',
        'LatEpicondyle',
        'MedEpicondyle',
    ]
    missing = [n for n in selectlandmarks if n not in landmarks]
    if missing:
        print(f'  WARNING: landmarks missing for plot: {", ".join(missing)}')
    landmarks_to_plot = {n: landmarks[n] for n in selectlandmarks if n in landmarks}
    _add_landmarks(plotter, landmarks_to_plot, arrow_scale)

    plotter.add_legend(
        [
            ['CT (scanner)', 'white'],
            ['Mako (knee center)', 'gold'],
            ['Cut post medial', 'cyan'],
            ['Cut dist medial', 'magenta'],
            ['landmarks', 'magenta'],
            ['bone STL', 'lightgrey'],
        ],
        bcolor='black',
    )
    plotter.add_axes()

    # Superior view: camera 200 mm above knee center (+Z), looking down (−Z)
    cam_height_mm = 200.0
    cam_pos = origin_mako_ct + np.array([0.0, 0.0, cam_height_mm])
    plotter.camera_position = [
        tuple(cam_pos),
        tuple(origin_mako_ct),
        (0.0, 1.0, 0.0),   # screen "up" = +Y (must not be parallel to view axis)
    ]

    plotter.show()
    # plotter.save_graphic(pfolder / f'Fem_csys_check.svg')


def read_asc(filepath: str) -> np.ndarray:
    """
    Read a .asc point file, skipping the first 3 header lines.
    Returns an (N, 3) float array.
    """
    points = []
    with open(filepath, 'r') as fid:
        for _ in range(3):          # skip header lines
            fid.readline()
        for ln in fid:
            ln = ln.strip()
            if len(ln) < 2:
                break
            vals = [v for v in ln.replace('\r', '').replace('\n', '').split(' ') if v]
            try:
                row = [float(v) for v in vals]
                if len(row) == 3:
                    points.append(row)
            except ValueError:
                continue
    return np.array(points)



def sort_contour(points: np.ndarray) -> np.ndarray:
    """
    Sort contour points along their boundary using nearest-neighbor traversal.
    
    Parameters
    ----------
    points : np.ndarray, shape (n, 3)
        Unsorted contour points.
    
    Returns
    -------
    sorted_points : np.ndarray, shape (n, 3)
        Points sorted along the contour boundary.
    """
    n = points.shape[0]
    visited = np.zeros(n, dtype=bool)
    sorted_idx = np.zeros(n, dtype=int)

    # Start from the point with minimum X (leftmost)
    current = np.argmin(points[:, 0])

    for i in range(n):
        sorted_idx[i] = current
        visited[current] = True

        # Find nearest unvisited neighbor
        diffs = points - points[current, :]
        dists = np.sum(diffs ** 2, axis=1)
        dists[visited] = np.inf  # exclude visited points

        if i < n - 1:
            current = np.argmin(dists)

    return points[sorted_idx]

def transform_mesh(mesh: pv.PolyData, transform: np.ndarray) -> pv.PolyData:
    """
    Transform a mesh using a 4x4 transformation matrix.
    """
    verts = mesh.points
    verts_h = np.hstack([verts, np.ones((len(verts), 1))]) # (N, 4) homogeneous
    verts_transformed = (transform @ verts_h.T).T[:, :3]              # (N, 3) in transformed frame
    mesh.points = verts_transformed
    return mesh

def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform points using a 4x4 transformation matrix.
    """
    points_h = np.hstack([points, np.ones((len(points), 1))]) # (N, 4) homogeneous
    points_transformed = (transform @ points_h.T).T[:, :3]              # (N, 3) in transformed frame
    return points_transformed


def detect_plane_level(mesh: pv.PolyData, axis: int, tol: float = 0.99,
                       bin_mm: float = 0.5) -> tuple:
    """
    Find planar facets whose normal is parallel to a coordinate axis and return
    the coordinate (along that axis) of the largest-area coplanar group.

    Useful for locating a resection plane on an implant component STL, e.g. the
    distal facet (normal ~ Z) or the posterior facet (normal ~ Y).

    Parameters
    ----------
    mesh : pv.PolyData
    axis : int
        0 = X, 1 = Y, 2 = Z.
    tol : float
        Dot-product threshold for "parallel to axis" (0.99 ~ 8 deg).
    bin_mm : float
        Bin width (mm) used to group coplanar faces along `axis`.

    Returns
    -------
    level : float
        Coordinate of the dominant (largest-area) plane along `axis`.
    candidates : list[tuple[float, float]]
        All detected planes as (level, total_area), sorted by area descending.
        Returned for inspection so the dominant pick can be verified/overridden.
    """
    norm = mesh.compute_normals(cell_normals=True, point_normals=False,
                                auto_orient_normals=True, consistent_normals=True)
    normals = np.asarray(norm.cell_data['Normals'])
    centers = norm.cell_centers().points
    areas = np.asarray(
        norm.compute_cell_sizes(area=True, length=False, volume=False).cell_data['Area']
    )

    # Relax the parallelism threshold until at least one facet qualifies, so the
    # function never silently returns NaN on a slightly off-axis / coarse mesh.
    axis_dot = np.abs(normals[:, axis])
    for thr in (tol, 0.95, 0.9, 0.8, 0.7):
        mask = axis_dot > thr
        if mask.any():
            break

    coords = centers[mask, axis]
    face_area = areas[mask]

    candidates = []
    if coords.size:
        keys = np.round(coords / bin_mm).astype(int)
        for k in np.unique(keys):
            sel = keys == k
            candidates.append((float(coords[sel].mean()), float(face_area[sel].sum())))
        candidates.sort(key=lambda c: c[1], reverse=True)
        level = candidates[0][0]
    else:
        level = float('nan')

    return level, candidates


def clip_keep_larger(mesh: pv.PolyData, normal, origin) -> pv.PolyData:
    """
    Clip a mesh with a plane and keep whichever side has more points.

    Removing a thin resection slab leaves a large remaining bone and a small
    discarded slab; keeping the larger side returns the bone regardless of the
    plane-normal sign convention.
    """
    side_a = mesh.clip(normal=normal, origin=origin, invert=False)
    side_b = mesh.clip(normal=normal, origin=origin, invert=True)
    return side_a if side_a.n_points >= side_b.n_points else side_b

# ── PATHS ──────────────────────────────────────────────────────────────────
study     = Path(r'S:\BiomechanicsResearch\groupImhauser\OREF TKA\Modeling')
data_raw  = study / 'Data_Raw' 
data_reduced = study / 'Data_Reduced'

subjects = ['S026']
isright = 1

for subject in subjects:
    model_inputs = data_reduced / subject / 'model_inputs'

    # check if the subject folder exists
    if not model_inputs.exists():
        print(f"Subject {subject} not found")
        continue

    screenshots = model_inputs / 'Screenshots'
    # check if the Screenshots folder exists
    if not screenshots.exists():
        print(f"Screenshots for subject {subject} not found")
        continue
    
    matches = [
        f for f in screenshots.iterdir()
        if '019_CaseInformation' in f.name and 'Anatomy' in f.name
    ]

    print(matches)
    if len(matches) == 0:
        print(f"No CaseInformation and Anatomy screenshots found for subject {subject}")
        continue
    
    # get the first match
    F9screenshot = matches[0]
    

    # Upscale 4x — ROI pixels were measured on the 4x image (7680 x 4320).
    img = np.array(Image.open(F9screenshot))
    h, w = img.shape[:2]
    img_4x = np.array(
        Image.fromarray(img).resize((w * 4, h * 4), Image.LANCZOS)
    )

    # --- Run reader ---
    planned = read_mako_planned_plan(img_4x, isright=isright)
    
    if verbose:
        # --- Display results ---
        print("\nPlanned summary:")
        print(f"  Component Size Femur     : {planned['component_size_femur']}")
        print(f"  Component Size Baseplate : {planned['component_size_baseplate']}")
        print(f"  Component Size Insert    : {planned['component_size_insert']} mm")
        print(f"  Planned Alignment        : {planned['planned_alignment']} "
            f"(+varus / -valgus)")
        print(f"  Planned Laxity Extension C1 : {planned['planned_laxity_extension_c1']} mm")
        print(f"  Planned Laxity Flexion C1   : {planned['planned_laxity_flexion_c1']} mm")
        print(f"  Planned Laxity Extension C2 : {planned['planned_laxity_extension_c2']} mm")
        print(f"  Planned Laxity Flexion C2   : {planned['planned_laxity_flexion_c2']} mm")
        print(f"  Femoral Rotation Coronal   : {planned['femoral_rotation_coronal']} °")
        print(f"  Femoral Rotation Transverse : {planned['femoral_rotation_transverse']} °")
        print(f"  Femoral Rotation Sagittal   : {planned['femoral_rotation_sagittal']} °")
        print(f"  Postslope Medial          : {planned['postslope_medial']} °")
        print(f"  Postslope Lateral         : {planned['postslope_lateral']} °")
        print(f"  Femoral Dist Resect Lateral    : {planned['femoral_dist_resect_lateral']} mm")
        print(f"  Femoral Dist Resect Medial    : {planned['femoral_dist_resect_medial']} mm")
        print(f"  Femoral Post Resect Lateral    : {planned['femoral_post_resect_lateral']} mm")
        print(f"  Femoral Post Resect Medial    : {planned['femoral_post_resect_medial']} mm")
        print(f"  Tibial Rotation Coronal    : {planned['tibial_rotation_coronal']} °")
        print(f"  Tibial Rotation Transverse : {planned['tibial_rotation_transverse']} °")
        print(f"  Tibial Rotation Sagittal    : {planned['tibial_rotation_sagittal']} °")
        print(f"  Tibial Resect Lateral    : {planned['tibial_resect_lateral']} mm")
        print(f"  Tibial Resect Medial    : {planned['tibial_resect_medial']} mm")
        print(f"  Planned Laxity Extension Lateral    : {planned['planned_laxity_extension_lateral']} mm")
        print(f"  Planned Laxity Extension Medial    : {planned['planned_laxity_extension_medial']} mm")
        print(f"  Planned Laxity Flexion Lateral    : {planned['planned_laxity_flexion_lateral']} mm")
        print(f"  Planned Laxity Flexion Medial    : {planned['planned_laxity_flexion_medial']} mm")
        # --- Visualize ROI boxes with OCR results ---
        print("\nGenerating ROI visualization...")
        vis = visualize_planned_rois(img_4x, save_path = model_inputs / 'Screenshots' / f"planned_roi_annotated_{subject}.jpg")
        vis.show()

    # read MAKO Points from CT
    mako_points = read_mako_points(model_inputs / 'MakoPoints'/ 'MakoPoints.txt')
    mako_points['AnkleCenter'] = (0.44 * mako_points['LatMalleolus'] +
                              0.56 * mako_points['MedMalleolus'])


    # ── BUILD COORDINATE SYSTEMS - Tibia ────────────────────────────────────────────────────
    tibia_plan = {
        'slope'     : planned['tibial_rotation_sagittal'],   # sagittal rotation  (col index 2)
        'varus'     : planned['tibial_rotation_coronal'],   # coronal rotation   (col index 0)
        'internal'  : planned['tibial_rotation_transverse'],   # axial rotation     (col index 1)
        'medialcut' : planned['tibial_resect_medial'],   # first cut value    (col index 3)
        'size'      : planned['component_size_baseplate'],   # implant size       (col index 7)
    }
    fem_plan = {
        'flexion'      : planned['femoral_rotation_sagittal'],   # sagittal rotation  (col index 2)
        'varus'     : planned['femoral_rotation_coronal'],   # coronal rotation   (col index 0)
        'internal'  : planned['femoral_rotation_transverse'],   # axial rotation     (col index 1)
        'medialdistcut' : planned['femoral_dist_resect_medial'],   # first cut value    (col index 3)
        'medialpostcut' : planned['femoral_post_resect_medial'],   # first cut value    (col index 3)
        'size'      : planned['component_size_baseplate'],   # implant size       (col index 7)
    }


    # Build coordinate systems — CT → Mako → cut plane
    # side is bool: True = right, False = left
    tibcsys, tiblandmarks_mako = tib_csys(mako_points, tibia_plan, isright)
    femcsys, femlandmarks_mako = fem_csys(mako_points, fem_plan, isright)


    
    # check_tibcsys(model_inputs / 'CT_data', subject, tibcsys, mako_points)
    # check_femcsys(model_inputs / 'CT_data', subject, femcsys, mako_points)


    # Build 4x4 homogeneous transform matrices from csys
    TibCT2MAKO = np.eye(4)
    TibCT2MAKO[:3, :3] = tibcsys['Rmako']
    TibCT2MAKO[:3,  3] = tibcsys['Tmako'].ravel()
    TibMAKO2CUT = np.eye(4)
    TibMAKO2CUT[:3, :3] = tibcsys['Rcut']
    TibMAKO2CUT[:3,  3] = tibcsys['Tcut'].ravel()
    TibCT2CUT = TibMAKO2CUT @ TibCT2MAKO



    FemCT2MAKO = np.eye(4)
    FemCT2MAKO[:3, :3] = femcsys['Rmako']
    FemCT2MAKO[:3,  3] = femcsys['Tmako'].ravel()
    FemMAKO2PCUT = np.eye(4)
    FemMAKO2PCUT[:3, :3] = femcsys['Rcut']
    FemMAKO2PCUT[:3,  3] = femcsys['TcutPostMed'].ravel()
    FemCT2PCUT = FemMAKO2PCUT @ FemCT2MAKO
    FemMAKO2DCUT = np.eye(4)
    FemMAKO2DCUT[:3, :3] = femcsys['Rcut']
    FemMAKO2DCUT[:3,  3] = femcsys['TcutDistMed'].ravel()
    FemCT2DCUT = FemMAKO2DCUT @ FemCT2MAKO






    # --------------------------- Resect and Align Tibia ---------------------------
    # Tibia
    # Get 
    tib_stl_path  = model_inputs / 'CT_data' / f'Tib.stl'
    tib_mesh = pv.read(str(tib_stl_path))
    tib_mesh_cut = transform_mesh(tib_mesh.copy(), TibCT2CUT)

    
    # ── 8B  RESECT: KEEP EVERYTHING BELOW Z=0 (the cut plane) ────────────────────
    # clip() keeps the side pointed to by 'normal' by default (invert=True flips it)
    tib_mesh_resected_cut = tib_mesh_cut.clip(
        normal = (0, 0, 1),    # cut plane normal = +Z in cut frame
        origin = (0, 0, 0),    # cut plane passes through origin
        invert = True          # keep Z < 0 (below the cut)
    )

    boundary_tibcut_cut = tib_mesh_resected_cut.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    # Tibia profile boundary
    ax.plot(boundary_tibcut_cut.points[:, 0], boundary_tibcut_cut.points[:, 1],
            'b.', markersize=2, label='Tibia profile')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.set_title(f'Tibia Cut profile — {subject}')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.tight_layout()
    # plt.savefig(model_inputs / 'Screenshots' / f'Tib_profile_{subject}.png', dpi=150)

    
    
    baseplate_size = int(planned['component_size_baseplate'])
    impcontour_path = study / 'Data_Raw' / 'Implants' / 'Triathlon_Cementless_3Dscans' / 'Tibial Tray Perimeters' / 'asc' / f'Triathlon_Ctless_Sz{baseplate_size}.asc'
    implant_contour = sort_contour(read_asc(impcontour_path))
    
    TibCUT2IMP = align_implant_to_cut(implant_contour, boundary_tibcut_cut.points, isright)
    # print(f'Bone-to-implant transform:\n{trm}')
    # write_tfm_file(trm, str(savefld), f'{pname}_CUT2IMP.tfm')
    # print(f'Saved: {pname}_CUT2IMP.tfm')
    fig, ax = plt.subplots(figsize=(8, 7))
 
    # Bone profile boundary
    ax.plot(boundary_tibcut_cut.points[:, 0], boundary_tibcut_cut.points[:, 1],
            'b.', markersize=2, label='Bone profile')
    
    # Initial implant position (centroid-aligned, before optimisation)
    ctr_cut     = boundary_tibcut_cut.points.mean(axis=0)
    ctr_implant = 0.5 * (implant_contour.max(axis=0) + implant_contour.min(axis=0))
    ax.plot(implant_contour[:, 0], implant_contour[:, 1],
            'r--', linewidth=1.5, label='Implant (initial)')
    
    # Aligned implant — apply inverse of trm to get display position
    impl_aligned = transform_points(implant_contour, np.linalg.inv(TibCUT2IMP))
    ax.plot(impl_aligned[:, 0], impl_aligned[:, 1],
            'g-', linewidth=2, label='Implant (aligned)')
    ax.plot(*ctr_cut,     'go', markersize=10, label='Bone centroid')
    ax.plot(*ctr_implant, 'rs', markersize=8, label='Implant centroid (shifted)')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.set_title(f'Implant alignment — {subject} (size {baseplate_size})')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.tight_layout()
    # plt.savefig(savefld / f'{pname}_implant_alignment.png', dpi=150)
    
    TibIMP2MAKO = np.linalg.inv(TibCUT2IMP@TibMAKO2CUT)

    tib_mesh_capped_cut = tib_mesh_resected_cut.fill_holes(hole_size=10000)
    tib_mesh_capped_imp = transform_mesh(tib_mesh_capped_cut.copy(), TibCUT2IMP)
    
    out_stl = model_inputs / 'CT_data' / f'{subject}_tib_resected_capped_cut.stl'
    tib_mesh_capped_imp.save(str(out_stl))
    print(f'Saved resected bone: {out_stl}')

    





    



    # Resect Bones Mesh
    # Femur
    fem_stl_path  = model_inputs / 'CT_data' / f'Fem.stl'
    fem_mesh = pv.read(str(fem_stl_path))
    # PCUT and DCUT share the same rotation (Rcut); only the origin differs, so
    # both cuts can be applied in the single DCUT frame. The posterior cut plane,
    # which is Y=0 in the PCUT frame, sits at Y=dy in the DCUT frame.
    dy = float((femcsys['TcutDistMed'].ravel() - femcsys['TcutPostMed'].ravel())[1])

    fem_mesh_dcut = transform_mesh(fem_mesh.copy(), FemCT2DCUT)

    # Apply both cuts, keeping the bulk femur each time. Unlike the tibia (whose
    # body is distal to its cut), the femur body is proximal/anterior to the cuts,
    # so the side to keep is the opposite one. clip_keep_larger picks the larger
    # fragment automatically, which is robust to the cut-plane normal sign.
    fem_mesh_resected = clip_keep_larger(fem_mesh_dcut, normal=(0, 0, 1), origin=(0, 0, 0))
    fem_mesh_resected = clip_keep_larger(fem_mesh_resected, normal=(0, 1, 0), origin=(0, dy, 0))
    
    # Distal-cut boundary profile (lies at Z~0 in the DCUT frame)
    boundary_dist = fem_mesh_resected.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False
    )

    # --------------------------- FemComp resection planes ---------------------------
    # The femoral component STL exposes a distal facet (normal ~ Z) and a
    # posterior facet (normal ~ Y). Extract their levels to seat the bone against
    # the implant. detect_plane_level returns the dominant (largest-area) plane
    # plus all candidates so the pick can be verified/overridden.
    femcomp_stl_path = model_inputs / 'CT_data' / f'FemComp.stl'
    femcomp_mesh = pv.read(str(femcomp_stl_path))

    z_dist, z_cands = detect_plane_level(femcomp_mesh, axis=2)   # distal plane (Z)
    y_post, y_cands = detect_plane_level(femcomp_mesh, axis=1)   # posterior plane (Y)
    if verbose:
        print(f'FemComp distal plane    Z = {z_dist:.3f}  (level, area) = {z_cands}')
        print(f'FemComp posterior plane Y = {y_post:.3f}  (level, area) = {y_cands}')

    # --------------------------- ML (X) alignment ---------------------------
    # Matching the two perpendicular planes fixes every DOF except the
    # medial-lateral (X) shift. Center the bone distal profile on the component
    # distal footprint in X.
    near_dist = np.abs(femcomp_mesh.points[:, 2] - z_dist) < 1.0
    comp_xc = 0.5 * (femcomp_mesh.points[near_dist, 0].max() +
                     femcomp_mesh.points[near_dist, 0].min())
    # The full femur boundary contains several open loops (posterior cut, proximal
    # end). Keep only the distal cut perimeter (Z ~ 0 in the DCUT frame).
    distal_pts = boundary_dist.points[np.abs(boundary_dist.points[:, 2]) < 1.0]
    bone_xc = 0.5 * (distal_pts[:, 0].max() + distal_pts[:, 0].min() )

    # --------------------------- Assemble DCUT -> FemComp transform ---------------------------
    # The DCUT axes already match the component axes (distal normal Z, posterior
    # normal Y), so the rotation is identity. If a render shows the bone seated on
    # the wrong side, replace R0 with a 180 deg rotation about X.
    R0 = np.eye(3)
    # R0 = np.diag([1.0, -1.0, -1.0])   # 180 deg about X (uncomment if flipped)

    # Anchor lying on both planes (distal Z=0, posterior Y=dy) at the bone X center
    src_anchor = np.array([bone_xc, dy, 0.0])
    dst_anchor = np.array([comp_xc, y_post, z_dist])

    FemCUT2COMP = np.eye(4)
    FemCUT2COMP[:3, :3] = R0
    FemCUT2COMP[:3,  3] = dst_anchor - R0 @ src_anchor

    fem_mesh_comp = transform_mesh(fem_mesh_resected.copy(), FemCUT2COMP)

    # --------------------------- Diagnostics ---------------------------
    if verbose:
        print(f'dy (posterior plane in DCUT)        = {dy:.3f}')
        print(f'bone_xc = {bone_xc:.3f}   comp_xc = {comp_xc:.3f}')
        print(f'FemCUT2COMP translation             = {FemCUT2COMP[:3, 3]}')
        print(f'fem_mesh_resected  n_points = {fem_mesh_resected.n_points}, '
              f'bounds = {np.round(fem_mesh_resected.bounds, 2)}')
        print(f'fem_mesh_comp      n_points = {fem_mesh_comp.n_points}, '
              f'bounds = {np.round(fem_mesh_comp.bounds, 2)}')
        print(f'femcomp_mesh       n_points = {femcomp_mesh.n_points}, '
              f'bounds = {np.round(femcomp_mesh.bounds, 2)}')
        if not np.all(np.isfinite(fem_mesh_comp.points)):
            print('WARNING: fem_mesh_comp contains non-finite points '
                  '(z_dist or y_post likely NaN -> no axis-aligned facet found).')

    # --------------------------- Verify ---------------------------
    # 2D check: bone distal boundary vs component distal footprint in the FemComp frame
    boundary_dist_comp = transform_points(distal_pts, FemCUT2COMP)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(boundary_dist_comp[:, 0], boundary_dist_comp[:, 1],
            'b.', markersize=2, label='Femur distal profile')
    ax.plot(femcomp_mesh.points[near_dist, 0], femcomp_mesh.points[near_dist, 1],
            'r.', markersize=2, label='FemComp distal footprint')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.set_title(f'Femur vs FemComp distal footprint — {subject}')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.tight_layout()

    # 3D check: overlay the aligned resected femur with the component
    plotter = pv.Plotter()
    plotter.add_mesh(fem_mesh_comp, color='lightgrey', opacity=0.7, label='resected femur')
    plotter.add_mesh(femcomp_mesh, color='lightblue', opacity=0.5, label='FemComp')
    plotter.add_legend()
    plotter.show()

    # --------------------------- Save ---------------------------
    fem_mesh_capped = fem_mesh_resected.fill_holes(hole_size=10000)
    fem_mesh_capped_comp = transform_mesh(fem_mesh_capped.copy(), FemCUT2COMP)
    out_stl = model_inputs / 'CT_data' / f'{subject}_fem_resected_capped_comp.stl'
    fem_mesh_capped_comp.save(str(out_stl))
    print(f'Saved resected femur: {out_stl}')

    plt.show()