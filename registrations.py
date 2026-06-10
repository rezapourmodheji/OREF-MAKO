
import numpy as np
import os
import re


# ── Helper function to create double-backslash path for Adams .cmd ────────────
def doublebs(path: str) -> str:
    """Convert single backslashes to double backslashes for Adams .cmd files."""
    return path.replace('\\', '\\\\')
def doublefs(path: str) -> str:
    """Convert single backslashes to double backslashes for Adams .cmd files."""
    print(f"Original path: {path}")
    return path.replace('/', '//')

def create_transformation_matrix(origin, x_axis, y_axis, z_axis):
    """
    Creates a 4×4 homogeneous transformation matrix from origin and orthonormal basis vectors.
    
    Parameters:
    -----------
    origin : np.ndarray
        Shape (3,) or (1,3) - position of the new frame origin in global coordinates
    x_axis : np.ndarray
        Shape (3,) or (1,3) - unit vector along the new frame's X-axis (in global coords)
    y_axis : np.ndarray
        Shape (3,) or (1,3) - unit vector along the new frame's Y-axis (in global coords)
    z_axis : np.ndarray
        Shape (3,) or (1,3) - unit vector along the new frame's Z-axis (in global coords)
    
    Returns:
    --------
    T : np.ndarray
        4×4 transformation matrix (rotation + translation)
        Format:
            [ R11 R12 R13 Tx ]
            [ R21 R22 R23 Ty ]
            [ R31 R32 R33 Tz ]
            [  0   0   0  1  ]
    """
    # Ensure all inputs are 1D arrays (flatten if needed)
    origin = np.asarray(origin).flatten()
    x_axis = np.asarray(x_axis).flatten()
    y_axis = np.asarray(y_axis).flatten()
    z_axis = np.asarray(z_axis).flatten()
    
    # Check shapes
    if any(arr.shape != (3,) for arr in [origin, x_axis, y_axis, z_axis]):
        raise ValueError("All inputs must be convertible to shape (3,)")

    # Build rotation matrix (columns are the basis vectors)
    R = np.column_stack((x_axis, y_axis, z_axis))
    
    # Create 4×4 transformation matrix
    T = np.eye(4)           # starts with identity
    T[:3, :3] = R           # insert rotation part
    T[:3, 3]  = origin      # insert translation part
    
    return T


def define_coordinate_system(p1, p2, p3) -> np.ndarray:
    """
    Define a 4x4 transformation matrix for the coordinate system.
    - Origin at p1
    - Y-axis along p1 to p2
    - XY-plane defined by p1, p2, p3
    Returns the 4x4 matrix T_local_to_global
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Y vector (normalized)
    vec_y = p2 - p1
    norm_y = vec_y / np.linalg.norm(vec_y)
    
    # Temporary vector for plane
    vec_temp = p3 - p1
    
    # Z vector: normal to the plane (cross product of vec_y and vec_temp)
    vec_z = np.cross(vec_y, vec_temp)
    norm_z = vec_z / np.linalg.norm(vec_z)
    
    # X vector: cross product of norm_y and norm_z for right-handed system
    # Wait, actually for right-handed: cross(norm_z, norm_y) or cross(norm_y, norm_z)?
    # To have cross(x, y) = z
    # x = cross(y, z)
    norm_x = np.cross(norm_y, norm_z)
    # Normalize (though should be unit already if inputs are)
    norm_x = norm_x / np.linalg.norm(norm_x)
    
    # Rotation matrix: columns are basis vectors x, y, z
    R = np.column_stack((norm_x, norm_y, norm_z))
    
    # 4x4 transformation: [R, origin; 0 0 0 1]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p1
    
    return T

def tfm2euler123(T: np.ndarray, degrees=True) -> np.ndarray:
    
    """
    Convert a 4x4 homogeneous transform to
    [x, y, z, alpha_x, beta_y, gamma_z]
    using intrinsic Euler 1-2-3 (Rx Ry Rz).
    """

    # --- Translation ---
    xyz = (T @ np.array([0, 0, 0, 1]))[:3]   # homogeneous multiplication
    x, y, z = xyz
    # print(f"Translation: x={x:.3f}, y={y:.3f}, z={z:.3f}")

    # --- Rotation matrix ---
    R = T[0:3, 0:3]

    # --- Euler 1-2-3 extraction (MATLAB equivalent) ---
    alpha = np.arctan2(-R[1, 2], R[2, 2])    # X
    beta  = np.arctan2( R[0, 2],
                         np.sqrt(R[0, 1]**2 + R[0, 0]**2) )  # Y
    gamma = np.arctan2(-R[0, 1], R[0, 0])    # Z

    angles = np.array([alpha, beta, gamma])

    if degrees:
        angles = np.degrees(angles)

    XYZ123 = np.hstack((x, y, z, angles))

    return XYZ123
    

def euler1232tfm(XYZ123: np.ndarray, degrees=True) -> np.ndarray:
    """
    Convert [x, y, z, alpha_x, beta_y, gamma_z]
    to a 4x4 homogeneous transform using
    intrinsic Euler 1-2-3 (Rx Ry Rz).

    This is the inverse of tfm2euler123().
    """

    x, y, z, alpha, beta, gamma = XYZ123

    if degrees:
        alpha = np.deg2rad(alpha)
        beta  = np.deg2rad(beta)
        gamma = np.deg2rad(gamma)

    # --- Rotation matrices ---
    cx, sx = np.cos(alpha), np.sin(alpha)
    cy, sy = np.cos(beta),  np.sin(beta)
    cz, sz = np.cos(gamma), np.sin(gamma)

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx,  cx]
    ])

    Ry = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy]
    ])

    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [ 0,   0, 1]
    ])

    # --- IMPORTANT: must match your extraction ---
    R = Rx @ Ry @ Rz

    # --- Assemble homogeneous transform ---
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z])

    return T

def euler3132tfm(XYZ313: np.ndarray, degrees=True) -> np.ndarray:
    """
    Convert [x, y, z, psi, theta, phi]
    to a 4x4 homogeneous transform using
    intrinsic Euler 3-1-3 (Rz Rx Rz).

    Angles:
        psi   : first rotation about Z
        theta : second rotation about new X
        phi   : third rotation about new Z
    """

    x, y, z, psi, theta, phi = XYZ313

    if degrees:
        psi   = np.deg2rad(psi)
        theta = np.deg2rad(theta)
        phi   = np.deg2rad(phi)

    # --- Rotation matrices ---
    cz1, sz1 = np.cos(psi), np.sin(psi)
    cx,  sx  = np.cos(theta), np.sin(theta)
    cz2, sz2 = np.cos(phi), np.sin(phi)

    Rz_psi = np.array([
        [cz1, -sz1, 0],
        [sz1,  cz1, 0],
        [  0,    0, 1]
    ])

    Rx_theta = np.array([
        [1,  0,   0],
        [0, cx, -sx],
        [0, sx,  cx]
    ])

    Rz_phi = np.array([
        [cz2, -sz2, 0],
        [sz2,  cz2, 0],
        [  0,    0, 1]
    ])

    # --- Intrinsic 3-1-3 ---
    R = Rz_psi @ Rx_theta @ Rz_phi

    # --- Homogeneous transform ---
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z])

    return T

def write_tfm_file(patacs, save_folder, filename="transformation.tfm"):
    """
    Write a .tfm file containing:
    - 4x4 transformation matrix (patacs)
    - blank lines
    - Euler angles (XYZ123) + translation with comments

    Parameters:
    -----------
    patacs : np.ndarray (4,4)
        The 4x4 homogeneous transformation matrix
    save_folder : str
        Directory where the file should be saved
    filename : str, optional
        Name of the output file (default: 'PatCT2PC.tfm')
    """
    # Make sure it's a numpy array and 4x4
    patacs = np.asarray(patacs)
    if patacs.shape != (4, 4):
        raise ValueError("patacs must be a 4×4 matrix")

    # Full path
    full_path = os.path.join(save_folder, filename)

    # Get the 6-element vector [rx, ry, rz, tx, ty, tz]
    # (assuming you already have the function from before)
    angles = tfm2euler123(patacs)

    # Open file in write mode
    with open(full_path, 'w', encoding='utf-8') as fid:
        # Write the 4×4 matrix (4 lines, space-separated)
        for row in patacs:
            fid.write(f"{row[0]:f} {row[1]:f} {row[2]:f} {row[3]:f}\n")

        # Two blank lines (like your \n\n)
        fid.write("\n\n")

        # Write the angles section
        fid.write("\n")
        fid.write("Units: mm\n")
        fid.write(f"# rotation x:   {angles[0]:10.6f}\n")
        fid.write(f"# rotation y:   {angles[1]:10.6f}\n")
        fid.write(f"# rotation z:   {angles[2]:10.6f}\n")
        fid.write(f"# translation x: {angles[3]:10.6f}\n")
        fid.write(f"# translation y: {angles[4]:10.6f}\n")
        fid.write(f"# translation z: {angles[5]:10.6f}\n")


def read_tfm_file(filepath):
    """
    Read a .tfm file and extract:
    - 4×4 homogeneous transformation matrix
    - Euler angles + translation vector (if present in comments)

    Returns
    -------
    patacs : np.ndarray (4,4)
        The 4×4 transformation matrix
    euler_xyz_t : np.ndarray (6,) or None
        [rx, ry, rz, tx, ty, tz] in degrees and mm (XYZ123 convention)
        or None if the comment section is missing / malformed

    Raises
    ------
    FileNotFoundError, ValueError
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    matrix_lines = []
    euler_section_found = False
    euler_values = [np.nan] * 6   # rx, ry, rz, tx, ty, tz

    with open(filepath, 'r', encoding='utf-8') as fid:
        lines = fid.readlines()

    # ── Phase 1: Read the 4×4 matrix (first 4 non-empty lines) ───────────────
    line_idx = 0
    while len(matrix_lines) < 4 and line_idx < len(lines):
        line = lines[line_idx].strip()
        line_idx += 1
        if not line or line.startswith('#') or line.startswith('!'):
            continue
        # Expect 4 floats per line
        try:
            vals = [float(x) for x in re.split(r'\s+', line) if x.strip()]
            if len(vals) == 4:
                matrix_lines.append(vals)
            else:
                raise ValueError(f"Expected 4 values per row, got {len(vals)}")
        except ValueError as e:
            raise ValueError(f"Failed to parse matrix row {len(matrix_lines)+1}: {line.strip()}") from e

    if len(matrix_lines) != 4:
        raise ValueError(f"Could not read exactly 4 rows for the transformation matrix (found {len(matrix_lines)})")

    patacs = np.array(matrix_lines, dtype=float)
    if patacs.shape != (4, 4):
        raise ValueError(f"Matrix has wrong shape: {patacs.shape}")

    # ── Phase 2: Look for Euler angles / translation in comment lines ────────
    for line in lines[line_idx:]:  # continue from where matrix ended
        line = line.strip()
        if not line or line.startswith('!'):
            continue

        # Match lines like: "# rotation x:   -12.345678"
        match = re.match(r'#\s*(rotation|translation)\s*([xyz])\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line, re.IGNORECASE)
        if match:
            euler_section_found = True
            what, axis, value_str = match.groups()
            value = float(value_str)

            if what.lower().startswith('rotation'):
                if axis.lower() == 'x':    euler_values[0] = value
                elif axis.lower() == 'y':  euler_values[1] = value
                elif axis.lower() == 'z':  euler_values[2] = value
            elif what.lower().startswith('translation'):
                if axis.lower() == 'x':    euler_values[3] = value
                elif axis.lower() == 'y':  euler_values[4] = value
                elif axis.lower() == 'z':  euler_values[5] = value

    if euler_section_found:
        euler_xyz_t = np.array(euler_values)
        if np.any(np.isnan(euler_xyz_t)):
            print("Warning: some Euler/translation values were not found → returning partial / NaN-filled array")
    else:
        euler_xyz_t = None
        print("Note: no Euler angles / translation comment section found in the file")

    return patacs, euler_xyz_t


