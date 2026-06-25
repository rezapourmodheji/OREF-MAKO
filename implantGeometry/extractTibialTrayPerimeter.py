"""
Extract tibial tray outer perimeters directly from STEP files.

This script reads STEP/STP implant files using CadQuery/OpenCascade,
extracts the outer wire of the largest planar tray face, orders the CAD
edges into a continuous perimeter loop, resamples the outline, and saves
the result as ASC coordinate files.

This avoids SolidWorks COM automation and is intended for batch processing
large folders of implant geometries.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from cadquery import importers


# ----------------------------------------------------
# PATHS
# ----------------------------------------------------

INPUT_DIR = Path(
    r"S:\BiomechanicsResearch\groupImhauser\OREF TKA\Modeling\Data_Raw\Implants\Triathlon_Cementless_3Dscans\Tibial Tray Perimeters\Tibial Tray Files"
)

OUT_DIR = INPUT_DIR.parent / "auto_step_perimeter_outerwire_output"
ASC_DIR = OUT_DIR / "asc"
FIG_DIR = OUT_DIR / "figures"

ASC_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------

N_OUTPUT_POINTS = 500

# Smaller = denser sampling of each CAD edge before final resampling.
EDGE_SAMPLE_SPACING_MM = 0.15

# Endpoint tolerance for connecting CAD edges into one continuous loop.
EDGE_CONNECT_TOLERANCE_MM = 0.25

# The selected tray face should be a large planar face.
MIN_PLANAR_FACE_AREA_MM2 = 100.0

SHOW_FIGURES = True

# Usually leave this False. It keeps only the clean perimeter line in QC.
SHOW_ASC_POINTS = True


# ----------------------------------------------------
# BASIC HELPERS
# ----------------------------------------------------

def vec_to_xyz(v):
    """Convert CadQuery/OCC vector-like object to [x, y, z]."""
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        return [float(v.x), float(v.y), float(v.z)]

    if hasattr(v, "toTuple"):
        t = v.toTuple()
        return [float(t[0]), float(t[1]), float(t[2])]

    if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 3:
        return [float(v[0]), float(v[1]), float(v[2])]

    raise TypeError(f"Could not convert vector to xyz: {type(v)}")


def get_shape_faces(workplane):
    """Collect all faces from all imported STEP solids/shapes."""
    faces = []

    try:
        solids = workplane.solids().vals()
        for solid in solids:
            faces.extend(solid.Faces())
    except Exception:
        pass

    if not faces:
        try:
            shapes = workplane.vals()
            for shape in shapes:
                faces.extend(shape.Faces())
        except Exception:
            pass

    if not faces:
        try:
            faces.extend(workplane.val().Faces())
        except Exception:
            pass

    return faces


def face_is_planar(face):
    try:
        return str(face.geomType()).upper() == "PLANE"
    except Exception:
        return False


def safe_face_area(face):
    try:
        return float(face.Area())
    except Exception:
        return 0.0


# ----------------------------------------------------
# FACE AND OUTER WIRE EXTRACTION
# ----------------------------------------------------

def find_largest_planar_face(faces):
    """
    Select the largest planar face.
    For these tibial trays, this should correspond to the main tray plate face.
    """
    planar_faces = []

    for face in faces:
        if not face_is_planar(face):
            continue

        area = safe_face_area(face)

        if area >= MIN_PLANAR_FACE_AREA_MM2:
            planar_faces.append((area, face))

    if not planar_faces:
        raise RuntimeError(
            "No sufficiently large planar faces found. "
            "Try lowering MIN_PLANAR_FACE_AREA_MM2."
        )

    planar_faces = sorted(planar_faces, key=lambda item: item[0], reverse=True)

    print("  Largest planar face candidates:")
    for i, (area, _) in enumerate(planar_faces[:5], start=1):
        print(f"    {i}: area = {area:.2f} mm^2")

    selected_area, selected_face = planar_faces[0]
    print(f"  Selected largest planar face area: {selected_area:.2f} mm^2")

    return selected_face


def get_outer_wire(face):
    """
    Get the true outer boundary wire of the selected planar face.
    Inner loops, screw holes, pegs, and keel features are ignored.
    """
    try:
        return face.outerWire()
    except Exception as e:
        raise RuntimeError(f"Could not extract outer wire from selected face: {e}")


def sample_edge(edge, spacing_mm):
    """
    Sample one CAD edge into a polyline of XYZ points.
    The sampled order may be forward or reversed, but the next function fixes continuity.
    """
    try:
        length = float(edge.Length())
    except Exception:
        length = spacing_mm * 10.0

    n = max(3, int(np.ceil(length / spacing_mm)) + 1)

    # Preferred CadQuery route.
    try:
        pts = edge.discretize(number=n)
        arr = np.array([vec_to_xyz(p) for p in pts], dtype=float)
        if len(arr) >= 2:
            return arr
    except Exception:
        pass

    # Fallback route.
    pts = []

    for t in np.linspace(0.0, 1.0, n):
        p = None

        for call in [
            lambda: edge.positionAt(t),
            lambda: edge.positionAt(t, mode="normalized"),
        ]:
            try:
                p = call()
                break
            except Exception:
                pass

        if p is not None:
            pts.append(vec_to_xyz(p))

    if len(pts) < 2:
        raise RuntimeError("Could not sample edge.")

    return np.array(pts, dtype=float)


def sample_outer_wire_edges(wire):
    """
    Sample all outer-wire edges independently.
    Do not connect them yet because CadQuery may return edges out of order.
    """
    try:
        edges = wire.Edges()
    except Exception as e:
        raise RuntimeError(f"Could not get edges from outer wire: {e}")

    if not edges:
        raise RuntimeError("Outer wire contains no edges.")

    print(f"  Outer wire edges: {len(edges)}")

    edge_polylines = []

    for edge in edges:
        pts = sample_edge(edge, EDGE_SAMPLE_SPACING_MM)

        # Remove local duplicate points within the edge.
        cleaned = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - cleaned[-1]) > 1e-9:
                cleaned.append(p)

        cleaned = np.array(cleaned, dtype=float)

        if len(cleaned) >= 2:
            edge_polylines.append(cleaned)

    if len(edge_polylines) < 2:
        raise RuntimeError("Too few usable edges extracted from outer wire.")

    return edge_polylines


# ----------------------------------------------------
# EDGE ORDERING
# ----------------------------------------------------

def cluster_endpoints(edge_polylines, tolerance):
    """
    Assign each edge endpoint to a cluster. Connected CAD edges should share clusters.
    """
    clusters = []
    start_cluster = []
    end_cluster = []

    def assign_cluster(point):
        for i, center in enumerate(clusters):
            if np.linalg.norm(point - center) <= tolerance:
                # Update cluster center slightly.
                clusters[i] = 0.5 * (center + point)
                return i

        clusters.append(point.copy())
        return len(clusters) - 1

    for pts in edge_polylines:
        s_id = assign_cluster(pts[0])
        e_id = assign_cluster(pts[-1])
        start_cluster.append(s_id)
        end_cluster.append(e_id)

    return np.array(start_cluster), np.array(end_cluster), np.array(clusters)


def order_edges_by_connectivity(edge_polylines):
    """
    Reorder and orient CAD edge polylines so they form one continuous perimeter loop.

    This is the key fix for the diagonal/chord continuity errors.
    """
    n_edges = len(edge_polylines)

    start_cluster, end_cluster, clusters = cluster_endpoints(
        edge_polylines,
        EDGE_CONNECT_TOLERANCE_MM,
    )

    # Print endpoint graph diagnostics.
    degrees = {}
    for s, e in zip(start_cluster, end_cluster):
        degrees[s] = degrees.get(s, 0) + 1
        degrees[e] = degrees.get(e, 0) + 1

    bad_clusters = [k for k, deg in degrees.items() if deg != 2]

    if bad_clusters:
        print(
            f"  WARNING: endpoint graph has {len(bad_clusters)} non-degree-2 nodes. "
            "Will use nearest-edge fallback if needed."
        )

    used = set()

    # Start with the longest edge. This makes traversal more stable than arbitrary edge 0.
    lengths = [
        np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        for pts in edge_polylines
    ]
    start_idx = int(np.argmax(lengths))

    ordered_segments = [edge_polylines[start_idx]]
    used.add(start_idx)

    current_cluster = end_cluster[start_idx]
    current_point = edge_polylines[start_idx][-1]

    max_connection_gap = 0.0

    while len(used) < n_edges:
        next_idx = None
        reverse_next = False

        # First try exact endpoint-cluster connectivity.
        for i in range(n_edges):
            if i in used:
                continue

            if start_cluster[i] == current_cluster:
                next_idx = i
                reverse_next = False
                break

            if end_cluster[i] == current_cluster:
                next_idx = i
                reverse_next = True
                break

        # Fallback: nearest unused edge endpoint.
        if next_idx is None:
            best_dist = np.inf
            best_idx = None
            best_reverse = False

            for i in range(n_edges):
                if i in used:
                    continue

                d_start = np.linalg.norm(edge_polylines[i][0] - current_point)
                d_end = np.linalg.norm(edge_polylines[i][-1] - current_point)

                if d_start < best_dist:
                    best_dist = d_start
                    best_idx = i
                    best_reverse = False

                if d_end < best_dist:
                    best_dist = d_end
                    best_idx = i
                    best_reverse = True

            next_idx = best_idx
            reverse_next = best_reverse

            print(
                f"  WARNING: nearest-edge fallback used. "
                f"Connection gap = {best_dist:.4f} mm"
            )

        pts = edge_polylines[next_idx]

        if reverse_next:
            pts = pts[::-1]
            next_current_cluster = start_cluster[next_idx]
        else:
            next_current_cluster = end_cluster[next_idx]

        gap = np.linalg.norm(pts[0] - current_point)
        max_connection_gap = max(max_connection_gap, gap)

        # Drop first point if it duplicates the last point in the growing chain.
        if gap <= EDGE_CONNECT_TOLERANCE_MM:
            pts_to_add = pts[1:]
        else:
            pts_to_add = pts

        ordered_segments.append(pts_to_add)
        used.add(next_idx)

        current_point = pts[-1]
        current_cluster = next_current_cluster

    ordered = np.vstack(ordered_segments)

    closure_gap = np.linalg.norm(ordered[-1] - ordered[0])

    print(f"  Max inter-edge connection gap: {max_connection_gap:.6f} mm")
    print(f"  Final loop closure gap:        {closure_gap:.6f} mm")

    if max_connection_gap > EDGE_CONNECT_TOLERANCE_MM:
        print(
            "  WARNING: Max connection gap exceeds tolerance. "
            "Inspect QC plot."
        )

    if closure_gap > EDGE_CONNECT_TOLERANCE_MM:
        print(
            "  WARNING: Loop closure gap exceeds tolerance. "
            "Inspect QC plot."
        )

    return ordered


# ----------------------------------------------------
# 3D TO 2D PROJECTION AND RESAMPLING
# ----------------------------------------------------

def project_boundary_to_2d(points_3d):
    """
    Project only the ordered outer perimeter into its best-fit 2D plane.
    Keel and internal features are never included.
    """
    center = points_3d.mean(axis=0)
    centered = points_3d - center

    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    u_axis = vt[0]
    v_axis = vt[1]

    x = centered @ u_axis
    y = centered @ v_axis

    pts_2d = np.column_stack([x, y])

    return pts_2d


def resample_closed_curve(points, n_points):
    """
    Resample an already ordered closed curve to evenly spaced ASC points.
    """
    pts = np.asarray(points, dtype=float)

    if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[0]])

    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)

    # Remove zero-length segments.
    keep_seg = seg_lengths > 1e-9

    cleaned_pts = [pts[0]]
    for i, keep in enumerate(keep_seg):
        if keep:
            cleaned_pts.append(pts[i + 1])

    pts = np.array(cleaned_pts, dtype=float)
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)

    s = np.insert(np.cumsum(seg_lengths), 0, 0.0)

    if s[-1] <= 0:
        raise RuntimeError("Curve length is zero after cleanup.")

    s_new = np.linspace(0, s[-1], n_points, endpoint=False)

    x_new = np.interp(s_new, s, pts[:, 0])
    y_new = np.interp(s_new, s, pts[:, 1])

    return np.column_stack([x_new, y_new])


# ----------------------------------------------------
# OUTPUT
# ----------------------------------------------------

def save_asc(path, points):
    with open(path, "w") as f:
        f.write("# X Y\n")
        for x, y in points:
            f.write(f"{x:.6f} {y:.6f}\n")


def save_qc_plot(path, raw_ordered_boundary, resampled_boundary, title):
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        raw_ordered_boundary[:, 0],
        raw_ordered_boundary[:, 1],
        linewidth=2.5,
        label="CAD outer wire, ordered",
    )

    if SHOW_ASC_POINTS:
        ax.scatter(
            resampled_boundary[:, 0],
            resampled_boundary[:, 1],
            s=6,
            alpha=0.65,
            label="Resampled ASC points",
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("Projected X (mm)")
    ax.set_ylabel("Projected Y (mm)")
    ax.legend()

    fig.savefig(path, dpi=400, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------------------------------
# MAIN PROCESSING
# ----------------------------------------------------

def process_file(step_path):
    print(f"\nProcessing: {step_path.name}")

    print("  Reading STEP with CadQuery/OpenCascade...")
    wp = importers.importStep(str(step_path))

    faces = get_shape_faces(wp)
    print(f"  Faces found: {len(faces)}")

    selected_face = find_largest_planar_face(faces)
    outer_wire = get_outer_wire(selected_face)

    edge_polylines = sample_outer_wire_edges(outer_wire)
    print(f"  Sampled edge polylines: {len(edge_polylines)}")

    ordered_boundary_xyz = order_edges_by_connectivity(edge_polylines)
    print(f"  Ordered raw boundary points: {len(ordered_boundary_xyz):,}")

    ordered_boundary_2d = project_boundary_to_2d(ordered_boundary_xyz)

    resampled_boundary = resample_closed_curve(
        ordered_boundary_2d,
        N_OUTPUT_POINTS,
    )

    asc_path = ASC_DIR / f"{step_path.stem}_outer_perimeter.asc"
    fig_path = FIG_DIR / f"{step_path.stem}_outer_perimeter_qc.png"

    save_asc(asc_path, resampled_boundary)
    save_qc_plot(
        fig_path,
        ordered_boundary_2d,
        resampled_boundary,
        step_path.stem,
    )

    print(f"  Saved ASC: {asc_path}")
    print(f"  Saved QC:  {fig_path}")


def main():
    print(f"Input directory:\n{INPUT_DIR}")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    files = sorted(
        {f.resolve() for ext in ["*.stp", "*.step"] for f in INPUT_DIR.glob(ext)}
    )

    print(f"\nFound {len(files)} STEP files:")
    for f in files:
        print(f"  {f.name}")

    for f in files:
        try:
            process_file(f)
        except Exception as e:
            print(f"  FAILED: {f.name}")
            print(f"  Error: {e}")

    print("\nDone.")
    print(f"ASC outputs: {ASC_DIR}")
    print(f"QC figures:  {FIG_DIR}")


if __name__ == "__main__":
    main()