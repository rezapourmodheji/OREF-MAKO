"""
contact_postprocess.py
======================
Mixin classes for postprocessing contact data from Adams knee simulations.
Add each mixin to AdamsKnee's base class list in core.py.
 
Stage 1  ContactGeometryMixin   — load + visualise STL geometries
Stage 2  (coming)               — parse contact incident files
Stage 3  ContactIncidentMixin   — load INSERT_XFORM tab file
Stage 4  ContactIncidentMixin   — compute dwell points & plot
"""
 
from pathlib import Path
import numpy as np
import pyvista as pv

 
# ---------------------------------------------------------------------------
# Component lists
# For PS (posterior-stabilised) inserts the tibial insert is split into parts;
# all other cases use a single monolithic Insert solid.
# ---------------------------------------------------------------------------
_STL_COMPONENTS_STANDARD = ["Insert", "Tib", "Fib", "Tray"]
_STL_COMPONENTS_PS       = ["InsertBase", "InsertMed", "InsertLat", "InsertPost",
                             "Tib", "Fib", "Tray"]
 
# Consistent colour scheme across all plots
_COMPONENT_COLORS = {
    "Insert":     "#e8d5b7",   # bone-white
    "InsertBase": "#e8d5b7",
    "InsertMed":  "#d4bfa0",
    "InsertLat":  "#c8b090",
    "InsertPost": "#bcaa85",
    "Tib":        "#b0c4de",   # steel-blue
    "Fib":        "#87a9c8",
    "Tray":       "#708090",   # slate-grey (metal tray)
}
 
 
# ===========================================================================
# Stage 1 — ContactGeometryMixin
# ===========================================================================
 
class ContactGeometryMixin:
    """
    Load and visualise STL geometry files for a given subject / case.
 
    Relies on paths already set by AdamsKnee.__init__:
        self.geom_dir   ->  .../Data_Reduced/Subjects/<sub>/<case>/<col>/model_inputs/Geometries/
        self.subject    ->  e.g. "TKAS11"
        self.case       ->  e.g. "CR", "MC", "PS"
 
    After calling load_stl_geometries() results are stored as:
        self.meshes : dict[str, pv.PolyData]
            keys   -> component name  ("Insert", "Tib", ...)
            values -> pyvista PolyData mesh
 
    Quick start
    -----------
        subcase = AdamsKnee("TKAS11", "CR", "L", "H", study)
        subcase.load_stl_geometries()          # loads & prints summary
        subcase.plot_geometries()              # interactive 3-D viewer
        pts = subcase.insert_vertices          # (N, 3) numpy array
    """
 
    # ------------------------------------------------------------------
    # Public: load
    # ------------------------------------------------------------------
 
    def load_stl_geometries(self, geom_dir=None, verbose=True):
        """
        Load STL files for this subject / case into self.meshes.
 
        Parameters
        ----------
        geom_dir : str or Path, optional
            Override the geometry directory. Defaults to self.geom_dir.
        verbose : bool
            Print a loading summary.
 
        Returns
        -------
        dict[str, pv.PolyData]  — same object stored at self.meshes
        """
        resolved_dir = self._resolve_geom_dir(geom_dir)
 
        components = (
            _STL_COMPONENTS_PS
            if getattr(self, "case", "").upper() == "PS"
            else _STL_COMPONENTS_STANDARD
        )
 
        if verbose:
            print(f"[{self.subject}] Loading STL geometries from:\n  {resolved_dir}")
 
        meshes = {}
        missing = []
 
        for name in components:
            stl_file = resolved_dir / f"{name}.stl"
            if not stl_file.exists():
                missing.append(name)
                if verbose:
                    print(f"  WARNING: {name}.stl not found — skipping.")
                continue
            try:
                mesh = pv.read(str(stl_file))
                # pyvista may return MultiBlock for some files; flatten to PolyData
                if isinstance(mesh, pv.MultiBlock):
                    mesh = mesh.combine().extract_surface()
                meshes[name] = mesh
                if verbose:
                    print(f"  {name:14s}: {mesh.n_points:6d} points, "
                          f"{mesh.n_cells:6d} faces")
            except Exception as exc:
                missing.append(name)
                if verbose:
                    print(f"  ERROR loading {name}.stl: {exc}")
 
        if missing and verbose:
            print(f"  Missing / failed: {missing}")
 
        self.meshes = meshes
        self._stl_geom_dir = resolved_dir   # re-used by later stages
        return meshes
 
    # ------------------------------------------------------------------
    # Public: visualise
    # ------------------------------------------------------------------
 
    def plot_geometries(
        self,
        components=None,
        opacity=None,
        show_edges=False,
        background="white",
        title=None,
        off_screen=False,
        screenshot=None,
    ):
        """
        Open an interactive pyvista viewer showing the loaded STL meshes.
 
        Parameters
        ----------
        components : list[str], optional
            Which components to show. Defaults to all loaded ones.
        opacity : dict[str, float] or float, optional
            Per-component opacity or a single value for all.
            Defaults to 0.85 for Insert parts, 1.0 for bone/tray.
        show_edges : bool
            Draw mesh wireframe edges.
        background : str
            Background colour (pyvista name or hex string).
        title : str, optional
            Window title.
        off_screen : bool
            Render off-screen (useful in scripts / CI).
        screenshot : str or Path, optional
            If given, save a PNG to this path instead of opening the window.
 
        Returns
        -------
        pv.Plotter
        """
        self._require_geometries()
 
        components = components or list(self.meshes.keys())
        title = title or f"{self.subject} — {self.case} geometries"
 
        pl = pv.Plotter(off_screen=off_screen, title=title)
        pl.set_background(background)
 
        for name in components:
            if name not in self.meshes:
                print(f"  plot_geometries: '{name}' not loaded, skipping.")
                continue
 
            color = _COMPONENT_COLORS.get(name, "#cccccc")
 
            # Resolve per-component opacity
            if isinstance(opacity, dict):
                op = opacity.get(name, 0.85 if "Insert" in name else 1.0)
            elif opacity is not None:
                op = float(opacity)
            else:
                op = 0.85 if "Insert" in name else 1.0
 
            pl.add_mesh(
                self.meshes[name],
                color=color,
                opacity=op,
                show_edges=show_edges,
                label=name,
                smooth_shading=True,
            )
 
        pl.add_legend(bcolor="white", border=True)
        pl.add_axes()
        pl.show_bounds(
            grid=False,
            location="outer",
            xlabel="ML (mm)",
            ylabel="AP (mm)",
            zlabel="SI (mm)",
            font_size=10,
        )
 
        if screenshot:
            pl.show(auto_close=False)
            pl.screenshot(str(screenshot))
            pl.close()
            print(f"  Screenshot saved -> {screenshot}")
        else:
            pl.show()
 
        return pl
 
    # ------------------------------------------------------------------
    # Public: convenience accessors
    # ------------------------------------------------------------------
 
    @property
    def insert_vertices(self) -> np.ndarray:
        """
        All tibial insert surface points as an (N, 3) numpy array.
        For PS cases, vertices from all Insert* parts are merged.
        """
        self._require_geometries()
        parts = [m for k, m in self.meshes.items() if "Insert" in k]
        if not parts:
            raise KeyError("No Insert mesh loaded. Check STL files.")
        if len(parts) == 1:
            return np.array(parts[0].points)
        combined = parts[0].merge(parts[1:])
        return np.array(combined.points)
 
    def get_mesh(self, component: str) -> pv.PolyData:
        """
        Return the pyvista PolyData for a named component.
 
        Parameters
        ----------
        component : str  e.g. "Insert", "Tib", "InsertMed"
        """
        self._require_geometries()
        if component not in self.meshes:
            raise KeyError(
                f"'{component}' not loaded. Available: {list(self.meshes)}"
            )
        return self.meshes[component]
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _resolve_geom_dir(self, geom_dir) -> Path:
        if geom_dir is not None:
            return Path(geom_dir)
        if hasattr(self, "geom_dir") and self.geom_dir is not None:
            return Path(self.geom_dir)
        if hasattr(self, "subject_dir") and self.subject_dir is not None:
            return Path(self.subject_dir) / "model_inputs" / "Geometries"
        raise RuntimeError(
            "Cannot resolve geometry directory. Either:\n"
            "  - Pass geom_dir=... to load_stl_geometries(), or\n"
            "  - Set self.geom_dir in AdamsKnee.__init__()."
        )
 
    def _require_geometries(self):
        if not hasattr(self, "meshes") or not self.meshes:
            raise AttributeError(
                "No meshes loaded. Call load_stl_geometries() first."
            )
 
 
# ===========================================================================
# Stage 2 + 3 — ContactIncidentMixin
# ===========================================================================
 
def _build_xform(X, Y, Z, PSI, THETA, PHI) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform from Adams Insert_XFORM Euler angles
    (degrees) and translation (mm).
 
    Replicates the MATLAB rotation construction exactly:
        R1 = [ sin(PHI)*sin(THETA)   cos(PHI)*sin(THETA)   cos(THETA)
               cos(PHI)             -sin(PHI)              0
               0                    0                      1          ]
        R2 = [ 0                    0                      1
               cos(PSI)             sin(PSI)               0
               sin(THETA)*sin(PSI) -sin(THETA)*cos(PSI)   cos(THETA) ]
        R  = inv(R2) @ R1
        T  = [ R | [X, Y, Z]^T
               0   1           ]
    """
    ph = np.deg2rad(PHI)
    th = np.deg2rad(THETA)
    ps = np.deg2rad(PSI)
 
    sp, cp = np.sin(ph), np.cos(ph)
    st, ct = np.sin(th), np.cos(th)
    ss, cs = np.sin(ps), np.cos(ps)
 
    R1 = np.array([
        [sp * st,  cp * st,  ct],
        [cp,      -sp,        0],
        [0,         0,        1],
    ])
    R2 = np.array([
        [0,       0,         1],
        [cs,      ss,        0],
        [st * ss, -st * cs,  ct],
    ])
    R = np.linalg.inv(R2) @ R1
 
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [X, Y, Z]
    return T
 
import re as _re
 
# Adams solid IDs for standard (non-PS) cases — mirrors the MATLAB constants:
#   InsertStr  = ['.' modelName '.Insert.SOLID3']
#   FemCompStr = ['.' modelName '.FemComp.SOLID2']
_FEMCOMP_SOLID = "FemComp.SOLID2"
_INSERT_SOLIDS_STANDARD = ["Insert.SOLID3"]
# PS inserts are split; Adams creates a separate contact surface per part.
# Any of these parts can appear in an incident block — we accept all of them.
_INSERT_SOLIDS_PS = [
    "InsertMed.SOLID4",
    "InsertLat.SOLID5"
]
 
 
def _parse_incident_file(filepath: Path, model_name: str,
                         insert_solids: list) -> dict:
    """
    Parse a single CONTACT_INCIDENT_<N>_<subject> file and return every
    contact event that involves the femoral component and any of the
    tibial insert solids.
 
    Parameters
    ----------
    filepath       : Path   — incident file (no extension)
    model_name     : str    — e.g. "TKAS11"
    insert_solids  : list   — solid suffixes to accept, e.g. ["Insert.SOLID3"]
                              or the PS list above
 
    Returns
    -------
    dict with keys:
        "i_points"     : (K, 3) float64 — I Point coords for K valid events
        "j_points"     : (K, 3) float64 — J Point coords
        "normal_forces": (K,)   float64 — normal force magnitudes
    Returns empty arrays if no valid events are found.
    """
    insert_tags = [f".{model_name}.{s}" for s in insert_solids]
    femcomp_tag = f".{model_name}.{_FEMCOMP_SOLID}"
    num_pattern = _re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
 
    text = filepath.read_text(encoding="utf-8", errors="replace")

    # Split on every "Contact Event" header (lookahead keeps the header in
    # each block, same logic as MATLAB's strfind + manual slicing).
    blocks = _re.split(r"(?=Contact Event)", text)
    i_points, j_points, normal_forces = [], [], []
    
    for block in blocks:
        # print('-------- Here Block ---------')
        # print(block)

        # Keep blocks that mention the femoral solid AND at least one insert solid
        if femcomp_tag not in block:
            # print('-------- Here 1 --------- :')
            # print(femcomp_tag)
            
            continue
        if not any(tag in block for tag in insert_tags):
            # print('-------- Here 2 --------- :')
            # print(insert_tags)
            
            continue
 
        lines = block.splitlines()
        
        i_line = next((l for l in lines if "I Point"     in l), None)
        j_line = next((l for l in lines if "J Point"     in l), None)
        f_line = next((l for l in lines if "Normal Force" in l), None)
        
        if not (i_line and j_line and f_line):
            continue
 
        i_nums = num_pattern.findall(i_line)
        j_nums = num_pattern.findall(j_line)
        f_nums = num_pattern.findall(f_line)
        
        # Last three numbers on each coordinate line are X, Y, Z
        # (Adams may prepend an event index number)
        if len(i_nums) < 3 or len(j_nums) < 3 or len(f_nums) < 1:
            continue
 
        i_points.append([float(v) for v in i_nums[-3:]])
        j_points.append([float(v) for v in j_nums[-3:]])
        normal_forces.append(float(f_nums[-1]))
    
    return {
        "i_points":      np.array(i_points,      dtype=np.float64).reshape(-1, 3),
        "j_points":      np.array(j_points,      dtype=np.float64).reshape(-1, 3),
        "normal_forces": np.array(normal_forces,  dtype=np.float64),
    }
 
 
class ContactIncidentMixin:
    """
    Stage 2: Parse saved contact incident files and store per-frame contact
    points and normal forces on the instance.
 
    Requires that the host class (AdamsKnee) exposes:
        self.subject      (str)   e.g. "TKAS11"
        self.contact_dir  (Path)  .../contact_incidents/
 
    The contact folder for a given simulation is named:
        <subject>_<test>_<compforce>_<pclcond>_<flex_angle>d/
 
    which matches exactly what _createcontactfolder() builds in simulation.py.
 
    Inside that folder:
        num_incidents              — plain text file: one integer = total frames
        CONTACT_INCIDENT_<N>_<subject>  — Adams list_info dump for frame N
        (frames are saved every 100 steps starting at step 20)
 
    After calling parse_contact_incidents() the results are stored as:
        self.contact_frames  : list[int]         — frame indices that were parsed
        self.contact_points  : list[np.ndarray]  — (K_m, 3) J-point coords per frame
        self.contact_ipoints : list[np.ndarray]  — (K_m, 3) I-point coords per frame
        self.contact_forces  : list[np.ndarray]  — (K_m,)   normal forces per frame
 
    The MATLAB script stores J Points as the contact location on the insert
    surface (cp_i) and I Points as the femoral side.  We follow the same
    convention: contact_points → J Points.
 
    Quick start
    -----------
        subcase = AdamsKnee("TKAS11", "CR", "L", "H", study)
        subcase.parse_contact_incidents(test="PassiveFlexion",
                                        compforce=10,
                                        pclcond="rPCL",
                                        flex_angle=90)
        # subcase.contact_points[m]  →  (K, 3) array of insert-side contact pts
        # subcase.contact_forces[m]  →  (K,)   normal forces for frame m
    """
 
    def parse_contact_incidents(
        self,
        test="PassiveFlexion",
        compforce=10,
        pclcond="rPCL",
        flex_angle=90,
        verbose=True,
    ):
        """
        Read all CONTACT_INCIDENT_* files for a simulation and parse contact
        points + normal forces.
 
        Parameters
        ----------
        test, compforce, pclcond, flex_angle
            Simulation identifiers — must match what was used in process_contact().
        verbose : bool
            Print a summary table.
 
        Returns
        -------
        dict with keys "frames", "contact_points", "contact_ipoints",
        "contact_forces"  (same data stored on self).
        """
        folder = self._resolve_contact_folder(test, compforce, pclcond, flex_angle)
 
        # ----------------------------------------------------------------
        # 1. Read num_incidents → total frame count (not the frame indices).
        #    The actual saved frame indices are reconstructed from the
        #    file naming pattern used in _write_contact():
        #        range(20, num_frames, 100)
        # ----------------------------------------------------------------
        num_file = folder / "num_incidents"
        if not num_file.exists():
            raise FileNotFoundError(
                f"num_incidents not found in:\n  {folder}\n"
                "Did process_contact() complete successfully?"
            )
        num_frames = int(num_file.read_text().strip())
        saved_frames = list(range(20, num_frames, 100))
 
        # PS inserts have four separate solids; all others use a single Insert
        insert_solids = (
            _INSERT_SOLIDS_PS
            if getattr(self, "case", "").upper() == "PS"
            else _INSERT_SOLIDS_STANDARD
        )
 
        if verbose:
            print(f"[{self.subject}] Parsing contact incidents — {folder.name}")
            print(f"  Insert solids : {insert_solids}")
            print(f"  Total frames  : {num_frames}")
            print(f"  Saved frames  : {len(saved_frames)}  "
                  f"(indices {saved_frames[0]}…{saved_frames[-1]})")
 
        # ----------------------------------------------------------------
        # 2. Parse each incident file
        # ----------------------------------------------------------------
        frames_parsed   = []
        contact_points  = []   # J Points  — insert surface side
        contact_ipoints = []   # I Points  — femoral side
        contact_forces  = []
 
        missing_files = []
 
        for frame_idx in saved_frames:
            inc_file = folder / f"CONTACT_INCIDENT_{int(frame_idx)}"
            if not inc_file.exists():
                missing_files.append(frame_idx)
                continue
 
            parsed = _parse_incident_file(inc_file, self.subject, insert_solids)
            
            if parsed["j_points"].size == 0:
                # No valid Insert↔FemComp events in this frame
                if verbose:
                    print(f"  WARNING: frame {frame_idx:5d} — no valid contact events found.")
                continue
 
            frames_parsed.append(frame_idx)
            contact_points.append(parsed["j_points"])
            contact_ipoints.append(parsed["i_points"])
            contact_forces.append(parsed["normal_forces"])
 
        if missing_files and verbose:
            print(f"  Missing incident files ({len(missing_files)}): "
                  f"{missing_files[:5]}{'…' if len(missing_files) > 5 else ''}")
 
        if verbose:
            total_events = sum(len(cp) for cp in contact_points)
            print(f"  Parsed frames: {len(frames_parsed)}  |  "
                  f"Total contact events: {total_events}")
        
        # ----------------------------------------------------------------
        # 3. Store on instance
        # ----------------------------------------------------------------
        self.contact_frames  = frames_parsed
        self.contact_points  = contact_points   # J Points (insert side)
        self.contact_ipoints = contact_ipoints  # I Points (femoral side)
        self.contact_forces  = contact_forces
 
        # Store the sim identifiers so later stages know which run this is
        self._contact_test       = test
        self._contact_compforce  = compforce
        self._contact_pclcond    = pclcond
        self._contact_flex_angle = flex_angle
        
        return {
            "frames":          frames_parsed,
            "contact_points":  contact_points,
            "contact_ipoints": contact_ipoints,
            "contact_forces":  contact_forces,
        }
 
    # ------------------------------------------------------------------
    # Convenience: quick sanity-check plot (2-D scatter, all frames)
    # ------------------------------------------------------------------
 
    def plot_contact_scatter(self, frame_indices=None, view="top"):
        """
        Quick matplotlib scatter of contact J-points as a sanity check.
        Mirrors the MATLAB scatter3 calls used for visual verification.
 
        Parameters
        ----------
        frame_indices : list[int], optional
            Which entries of self.contact_frames to plot.
            Defaults to all parsed frames.
        view : "top" | "ap" | "ml"
            Which 2-D projection to show:
            "top" → ML vs AP  (bird's eye, insert surface plane)
            "ap"  → ML vs SI
            "ml"  → AP vs SI
        """
        self._require_contact()
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
 
        indices = (
            list(range(len(self.contact_frames)))
            if frame_indices is None
            else frame_indices
        )
 
        axis_map = {
            "top": (0, 1, "ML (mm)", "AP (mm)"),
            "ap":  (0, 2, "ML (mm)", "SI (mm)"),
            "ml":  (1, 2, "AP (mm)", "SI (mm)"),
        }
        xi, yi, xlabel, ylabel = axis_map.get(view, axis_map["top"])
 
        colors = cm.viridis([i / max(len(indices), 1) for i in range(len(indices))])
 
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{self.subject} — {getattr(self, '_contact_pclcond', '')} "
            f"contact points ({view} view)"
        )
        ax.grid(True, linewidth=0.4)
 
        for color, idx in zip(colors, indices):
            pts = self.contact_points[idx]
            if pts.size == 0:
                continue
            ax.scatter(pts[:, xi], pts[:, yi], s=12, color=color, alpha=0.7)
 
        # Colourbar for frame progression
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(
                vmin=self.contact_frames[indices[0]],
                vmax=self.contact_frames[indices[-1]],
            ),
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Frame index")
        plt.tight_layout()
        plt.show()
        return fig, ax
 
    # ------------------------------------------------------------------
    # Stage 3: INSERT_XFORM — read per-frame insert pose from the .tab file
    # ------------------------------------------------------------------
 
    def load_insert_xform(
        self,
        test="PassiveFlexion",
        compforce=10,
        pclcond="rPCL",
        flex_angle=90,
        verbose=True,
    ):
        """
        Read Insert pose columns from the motion output .tab file and build
        a 4x4 homogeneous transform for every time step.
 
        For standard cases (non-PS) the columns are:
            Insert_XFORM.X / Y / Z / PSI / THETA / PHI
 
        For PS cases Adams exports the pose of each insert part separately.
        We use InsertBase_XFORM.* as the reference rigid body (the base plate
        that carries all four parts).
 
        The rotation convention follows the MATLAB script exactly:
            R1 = [ sin(PHI)*sin(THETA)   cos(PHI)*sin(THETA)   cos(THETA)
                   cos(PHI)             -sin(PHI)              0
                   0                    0                      1          ]
            R2 = [ 0                    0                      1
                   cos(PSI)             sin(PSI)               0
                   sin(THETA)*sin(PSI) -sin(THETA)*cos(PSI)   cos(THETA) ]
            R  = inv(R2) @ R1
            T  = [ R | [X, Y, Z]^T ]
                 [ 0   1           ]
 
        Results stored on instance
        --------------------------
        self.xform_time      : (F,)     float64 — time vector
        self.xform_matrices  : (F,4,4)  float64 — transform at each frame
        self.xform_df        : DataFrame         — raw tab columns (all measures)
 
        Returns
        -------
        xform_matrices : (F, 4, 4) numpy array
        """
        import pandas as pd
 
        output_name = (
            f"{self.subject}_{test}_{int(compforce)}_"
            f"{pclcond}_{int(flex_angle)}d"
        )
        tab_path = self.output_dir / f"{output_name}.tab"
        if not tab_path.exists():
            raise FileNotFoundError(
                f"Motion output tab file not found:\n  {tab_path}"
            )
 
        df = pd.read_csv(tab_path, sep="\t", skiprows=1)
        df.columns = [c.strip() for c in df.columns]

        
 
        # Choose the right XFORM prefix depending on implant type
        prefix = "InsertBase_XFORM" if getattr(self, "case", "").upper() == "PS" else "Insert_XFORM"
        
        
        required = [f"{prefix}.X", f"{prefix}.Y", f"{prefix}.Z",
                    f"{prefix}.PSI", f"{prefix}.THETA", f"{prefix}.PHI"]
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            raise KeyError(
                f"Expected columns not found in {tab_path.name}: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )
 
        X     = df[f"{prefix}.X"].values
        Y     = df[f"{prefix}.Y"].values
        Z     = df[f"{prefix}.Z"].values
        PSI   = df[f"{prefix}.PSI"].values
        THETA = df[f"{prefix}.THETA"].values
        PHI   = df[f"{prefix}.PHI"].values
 
        n = len(X)
        xform_matrices = np.zeros((n, 4, 4), dtype=np.float64)
 
        for i in range(n):
            xform_matrices[i] = _build_xform(
                X[i], Y[i], Z[i], PSI[i], THETA[i], PHI[i]
            )
 
        self.xform_time     = df["Time"].values if "Time" in df.columns else np.arange(n)
        self.xform_matrices = xform_matrices
        self.xform_df       = df
        self.xform_alpha    = df['Alpha'].tolist()
        self.xform_q2       = df['q2'].tolist()
 
        if verbose:
            print(f"[{self.subject}] Loaded INSERT_XFORM from: {tab_path.name}")
            print(f"  XFORM prefix : {prefix}")
            print(f"  Time steps   : {n}  ({self.xform_time[0]:.3f}s → {self.xform_time[-1]:.3f}s)")
 
        return xform_matrices
 
    # ------------------------------------------------------------------
    # Stage 4a: compute dwell points — transform contact pts to insert frame
    # ------------------------------------------------------------------
 
    def compute_dwell_points(self, verbose=True):
        """
        For every parsed contact frame, transform the most-posterior medial
        and lateral contact points from the global Adams frame into the
        insert-local coordinate frame.
 
        Replicates the MATLAB dwell-point loop exactly:
          1. Retrieve the insert XFORM at the frame's time-step index
             (nearest-neighbour lookup into self.xform_time using the
             frame index, which maps to a time via step_size = 0.01 s)
          2. Split contact points into medial (ML col > 0) and
             lateral (ML col < 0) — column index 1 in global coords
          3. Pick the most-posterior point on each side:
             min value along the AP/CD axis (column index 0)
          4. Apply  inv(XFORM) @ [pt; 1]  to get insert-local coords
 
        Requires
        --------
        parse_contact_incidents() and load_insert_xform() must have been
        called first.
 
        Results stored on instance
        --------------------------
        self.dwell_medial  : list[(3,) ndarray]  — one per parsed frame
        self.dwell_lateral : list[(3,) ndarray]  — one per parsed frame
        self.dwell_frames  : list[int]           — frame indices (subset of
                                                   contact_frames that had
                                                   both med & lat points)
 
        Returns
        -------
        dict with keys "frames", "medial", "lateral"
        """
        self._require_contact()
        self._require_xform()
 
        dwell_medial    = []
        dwell_lateral   = []
        cforce_medial   = []
        cforce_lateral  = []
        dwell_frames    = []
        skipped         = []
 
        # # Frame index → time:  Adams uses step_size = 0.01 s
        # step_size = 0.01

        for m, frame_idx in enumerate(self.contact_frames):
            pts = self.contact_points[m]   # (K, 3) J-points in global frame
            forces = self.contact_forces[m]  # (K,)   normal forces for these points

            # ---- medial: ML (col 1) > 0 --------------------------------
            med_mask = pts[:, 1] > 0
            lat_mask = pts[:, 1] < 0
            
 
            if med_mask.sum() == 0 and lat_mask.sum() == 0:
                skipped.append(frame_idx)
                continue
            if med_mask.sum() == 0 or lat_mask.sum() == 0:
                if med_mask.sum() == 0:
                    print(f"  WARNING: frame {frame_idx} — no medial contact points found.")
                    med_local = np.array([np.nan, np.nan, np.nan])
                    med_force = 0.0
                if lat_mask.sum() == 0:
                    print(f"  WARNING: frame {frame_idx} — no lateral contact points found.")
                    lat_local = np.array([np.nan, np.nan, np.nan])
                    lat_force = 0.0

            
            else:
                # Most-posterior = minimum value along col 0 (AP/CD axis)
                med_pts = pts[med_mask]
                lat_pts = pts[lat_mask]
                med_forces = forces[med_mask]
                lat_forces = forces[lat_mask]

                med_pt  = med_pts[np.argmax(med_forces)]   # (3,)
                lat_pt  = lat_pts[np.argmax(lat_forces)]   # (3,)
                med_force = med_forces[np.argmax(med_forces)]
                lat_force = lat_forces[np.argmax(lat_forces)]
                
    
                # ---- look up XFORM at this frame's time --------------------
                # self.xform_time 
                # frame_time = frame_idx * step_size
                # xform_row  = np.argmin(np.abs(self.xform_time - frame_time))
                T          = self.xform_matrices[frame_idx]    # (4,4)
                T_inv      = np.linalg.inv(T)
    
                # ---- transform to insert-local frame -----------------------
                med_local = (T_inv @ np.append(med_pt, 1.0))[:3]
                lat_local = (T_inv @ np.append(lat_pt, 1.0))[:3]

                
            
            dwell_medial.append(med_local)
            dwell_lateral.append(lat_local)
            dwell_frames.append(frame_idx)
            cforce_medial.append(med_force)
            cforce_lateral.append(lat_force)
            
        
        self.dwell_medial  = dwell_medial
        self.dwell_lateral = dwell_lateral
        self.dwell_frames  = dwell_frames
        self.cforce_medial = cforce_medial
        self.cforce_lateral = cforce_lateral
        
        if verbose:
            print(f"[{self.subject}] Dwell points computed")
            print(f"  Frames with both compartments : {len(dwell_frames)}")
            if skipped:
                print(f"  Skipped (missing med or lat)  : {len(skipped)}  "
                      f"— frames {skipped[:5]}{'…' if len(skipped) > 5 else ''}")
 
        return {
            "frames":  dwell_frames,
            "medial":  dwell_medial,
            "lateral": dwell_lateral,
            "cforce_medial": cforce_medial,
            "cforce_lateral": cforce_lateral,
        }
 
    # ------------------------------------------------------------------
    # Stage 4b: plot dwell points overlaid on insert geometry
    # ------------------------------------------------------------------
 
    def plot_dwell_points(
        self,
        target_flexion_angles=None,
        screenshot=None,
        off_screen=False,
    ):
        """
        Plot dwell points overlaid on the insert vertex cloud (pyvista).
 
        Replicates the MATLAB scatter3 plot:
          • Insert vertices drawn as a small grey point cloud (backdrop)
          • For each target flexion angle, medial and lateral dwell points
            are drawn with distinct markers and colours:
              - 0°  medial  → red  diamond  (■)
              - 0°  lateral → red  square   (□)
              - 90° medial  → green circle  (●)
              - 90° lateral → green star    (★)
 
        Target frames are located by finding the contact frame whose
        time (frame_idx * 0.01 s) is nearest to t=2 s for 0° flexion,
        and the frame whose Alpha column value is nearest to the requested
        angle for non-zero targets.  This mirrors MATLAB's:
            ind00 = dsearchn(Time, 2)
            ind90 = dsearchn(Alpha, 90)
 
        Parameters
        ----------
        target_flexion_angles : list[float], optional
            Flexion angles to highlight.  Defaults to [0, 90].
            0° is always located by time (t≈2 s); non-zero angles are
            located by the nearest Alpha value in xform_df.
        screenshot : str or Path, optional
            If given, save a PNG instead of opening the interactive window.
        off_screen : bool
            Render off-screen (for scripts / CI).
 
        Requires
        --------
        load_stl_geometries(), parse_contact_incidents(),
        load_insert_xform(), and compute_dwell_points() must have been
        called first.
        """
        self._require_dwell()
 
        import pyvista as pv
        
        target_angles = target_flexion_angles if target_flexion_angles is not None else [0, 90]
 
        # Marker colours and shapes per (angle, compartment)
        # pyvista point glyphs: we use sphere/cube glyphs scaled by size
        style_map = {
            # (angle==0, medial)   → red,   (angle==0, lateral)   → red
            # (angle!=0, medial)   → green, (angle!=0, lateral)   → green
        }
        colors  = {0: "blue",   "other": "red"}
        # markers = {
        #     (True,  True):  ("diamond", 18),   # 0°  medial
        #     (True,  False): ("square",  18),   # 0°  lateral
        #     (False, True):  ("circle",  18),   # 90° medial
        #     (False, False): ("star",    18),   # 90° lateral
        # }
 
        title = (f"{self.subject} — {getattr(self, '_contact_pclcond', '')} "
                 f"dwell points")
        pl = pv.Plotter(off_screen=off_screen, title=title)
        pl.set_background("white")
 
        # ── Insert vertex cloud ──────────────────────────────────────────
        # if hasattr(self, "meshes") and "Insert" in self.meshes:
        #     insert_pts = pv.PolyData(np.array(self.meshes["Insert"].points))
        #     pl.add_mesh(insert_pts, color="lightgrey", point_size=3,
        #                 render_points_as_spheres=True, label="Insert")
        # elif hasattr(self, "meshes"):
        #     # PS: merge all Insert* parts
        #     all_pts = self.insert_vertices
        #     insert_cloud = pv.PolyData(all_pts)
        #     pl.add_mesh(insert_cloud, color="lightgrey", point_size=3,
        #                 render_points_as_spheres=True, label="Insert")
        plt_meshes = [pv.read(self.geom_dir / f"{name}.stl") for name in self.meshes if any(keyword in name for keyword in ['Insert']) ]
        for mesh in plt_meshes:
            pl.add_mesh(mesh, color='#FFFFF0', 
                        pbr=True, 
                        metallic=0.0, 
                        roughness=0.4,  # Slightly above 0 for a realistic shine
                        smooth_shading=True,
                        opacity = 1
            )
        
        noins_meshes =  [pv.read(self.geom_dir / f"{name}.stl") for name in self.meshes if any(keyword in name for keyword in ['Tib', 'Fib']) ]
        for mesh in noins_meshes:
            pl.add_mesh(mesh, color='#E3DAC9', 
                        pbr=True, 
                        metallic=0.0, 
                        roughness=0.4,  # Slightly above 0 for a realistic shine
                        smooth_shading=True
            )
        # ── Dwell points per target angle ────────────────────────────────
        frames_arr = np.array(self.dwell_frames)
        # step_size  = 0.01
        # self.xform_time[] 
        

        for angle in target_angles:
            is_zero = (angle == 0)
 
            if is_zero:
                # Find dwell frame whose time is nearest to t = 2 s
                # (MATLAB: ind00 = dsearchn(Time, 2))
                frame_times = np.array([self.xform_time[i] for i in self.dwell_frames]) #frames_arr * step_size
                dwell_idx   = int(np.argmin(np.abs(frame_times - 2.0)))
            else:
                # Find dwell frame whose flexion angle is nearest to target
                # (MATLAB: ind90 = dsearchn(Alpha, 90))
                if "Alpha" not in self.xform_df.columns:
                    print(f"  WARNING: 'Alpha' column not in xform_df — "
                          f"cannot locate {angle}° frame. Skipping.")
                    continue
                alpha_vals  = self.xform_df["Alpha"].values
                # Map each dwell frame_idx → nearest row in xform_df
                target_row  = int(np.argmin(np.abs(alpha_vals - angle)))
                target_time = self.xform_time[target_row]
                frame_times = np.array([self.xform_time[i] for i in self.dwell_frames]) #frames_arr * step_size
                dwell_idx   = int(np.argmin(np.abs(frame_times - target_time)))
 
            med_pt = np.array(self.dwell_medial[dwell_idx]).reshape(1, 3)
            med_pt[:,0] += 4  # shift medial points slightly for better visibility in the plot
            lat_pt = np.array(self.dwell_lateral[dwell_idx]).reshape(1, 3)
            lat_pt[:,0] += 4  # shift lateral points slightly for better visibility in the plot
 
            color = colors[0] if is_zero else colors["other"]
            angle_label = f"{int(angle)}°"
 
            pl.add_mesh(
                pv.PolyData(med_pt),
                color=color, point_size=20,
                render_points_as_spheres=True,
                label=f"Med {angle_label}",
            )
            pl.add_mesh(
                pv.PolyData(lat_pt),
                color=color, point_size=20,
                render_points_as_spheres=True,
                label=f"Lat {angle_label}",
            )
            lines = np.array([med_pt[0], lat_pt[0]])
            pl.add_lines(lines, color=color, width=5)
            if off_screen:
                # Annotate with flexion angle label
                pl.add_point_labels(
                    med_pt, [f"Med {angle_label}"],
                    font_size=10, text_color=color,
                    always_visible=True, shadow=True,
                )
                pl.add_point_labels(
                    lat_pt, [f"Lat {angle_label}"],
                    font_size=10, text_color=color,
                    always_visible=True, shadow=True,
                )
 
        # pl.add_legend(bcolor="white", border=True)
        # pl.add_axes()
        # pl.show_bounds(
        #     grid=False, location="outer",
        #     xlabel="AP (mm)", ylabel="ML (mm)", zlabel="SI (mm)",
        #     font_size=10,
        # )
        pl.camera.position = (175, 0, 0.0)
        pl.camera.focal_point = (0, 0, 0)
        # pl.camera.up = (0.0, 0, 0.0)
        pl.camera.zoom(1.0)
        light = pv.Light(position=(100, 0, -25),  show_actor=False, positional=True,
                 cone_angle=30, exponent=10, intensity=2, color='white')
        pl.add_light(light)

        # if screenshot:
        #     pl.show(auto_close=False)
        #     pl.screenshot(str(screenshot))
        #     pl.close()
        #     print(f"  Screenshot saved → {screenshot}")
        # else:
        #     pl.show()
        # exit(1)
        return pl
 
    # ------------------------------------------------------------------
    # Stage 4c: export dwell points to a tidy DataFrame / CSV
    # ------------------------------------------------------------------
 
    def dwell_points_to_dataframe(self):
        """
        Return a tidy pandas DataFrame with one row per frame containing
        medial and lateral dwell point coordinates in insert-local frame.
 
        Columns: frame_idx, time_s,
                 med_x, med_y, med_z,
                 lat_x, lat_y, lat_z,
                 subject, case, pclcond
 
        Returns
        -------
        pd.DataFrame
        """
        import pandas as pd
        self._require_dwell()
 
        
        rows = []
        
        for i, frame_idx in enumerate(self.dwell_frames):
            med = self.dwell_medial[i]
            lat = self.dwell_lateral[i]
            cforce_med = self.cforce_medial[i]
            cforce_lat = self.cforce_lateral[i]
            rows.append({
                "subject":   self.subject,
                "case":      getattr(self, "case",             ""),
                "coltension":getattr(self, "coltension",       ""),
                "pclcond":   getattr(self, "_contact_pclcond", ""),
                "frame_idx": frame_idx,
                "time_s":    self.xform_time[frame_idx], #round(frame_idx * step_size, 4),
                "med_x": med[0], "med_y": med[1], "med_z": med[2],
                "lat_x": lat[0], "lat_y": lat[1], "lat_z": lat[2],
                "cforce_med": cforce_med,
                "cforce_lat": cforce_lat,
            })
        
        contact_df = pd.DataFrame(rows)
        return contact_df
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _resolve_contact_folder(self, test, compforce, pclcond, flex_angle) -> Path:
        """Build the contact folder path, mirroring _createcontactfolder()."""
        folder_name = f"{self.subject}_{test}_{compforce}_{pclcond}_{int(flex_angle)}d"
        folder = self.contact_dir / folder_name
        if not folder.exists():
            raise FileNotFoundError(
                f"Contact folder not found:\n  {folder}\n"
                "Run process_contact() first (or check test/compforce/pclcond/flex_angle)."
            )
        return folder
 
    def _require_contact(self):
        if not hasattr(self, "contact_frames") or not self.contact_frames:
            raise AttributeError(
                "No contact data loaded. Call parse_contact_incidents() first."
            )
 
    def _require_xform(self):
        if not hasattr(self, "xform_matrices") or self.xform_matrices is None:
            raise AttributeError(
                "No XFORM data loaded. Call load_insert_xform() first."
            )
 
    def _require_dwell(self):
        if not hasattr(self, "dwell_frames") or not self.dwell_frames:
            raise AttributeError(
                "No dwell points computed. Call compute_dwell_points() first."
            )