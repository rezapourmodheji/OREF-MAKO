from time import time
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
import imageio
from pathlib import Path
import seaborn as sns
sns.set()
from registrations import (
    create_transformation_matrix,
    euler1232tfm,
    euler3132tfm,
    tfm2euler123,
    write_tfm_file,
    read_tfm_file,
)

def transform_point(point: np.ndarray, tfm: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transformation matrix to a 3D point.
 
    Parameters
    ----------
    point : (3,) array  — point in its local (reference) frame
    tfm   : (4,4) array — homogeneous transform
 
    Returns
    -------
    (3,) array — point in the global frame
    """
    p_h = np.append(point, 1.0)      # homogeneous: [x, y, z, 1]
    return (tfm @ p_h)[:3]


def add_global_axes(plotter: pv.Plotter, scale: float = 30.0):
    """Draw RGB X-Y-Z arrows at the global origin."""
    origin = np.array([[0.0, 0.0, 0.0]])
    for label, vec, color in [
        ('X', np.array([[1, 0, 0]]), 'red'),
        ('Y', np.array([[0, 1, 0]]), 'green'),
        ('Z', np.array([[0, 0, 1]]), 'blue'),
    ]:
        plotter.add_arrows(origin, vec * scale, color=color)
        tip = (vec * scale * 1.15)[0]
        plotter.add_point_labels([tip], [label], font_size=14,
                                  text_color=color, bold=True,
                                  show_points=False, always_visible=True)
        

class ProcessMotionMixin:
    def measureGap(self,  test = 'CompValVar', compforce=10, pclcond='rPCL', flex_angle = 90,
                   render_video=True, verbose=False):
        """
        Quantify medial and lateral compartment gap during a varus-valgus test.
 
        Strategy
        --------
        Dwell points are defined in the LOCAL (reference) frame of each
        component — i.e. the frame in which the STL was originally built.
        At each time step we apply the rigid body transform to bring them
        into the global lab frame, then compute the Euclidean distance
        between paired points (femoral condyle → insert plateau).
 
        Dwell point pairs
        -----------------
          Medial  : insert_med  ↔  femcomp_distmed   (distal condyle)
                    insert_med  ↔  femcomp_postmed    (posterior condyle)
          Lateral : insert_lat  ↔  femcomp_distlat
                    insert_lat  ↔  femcomp_postlat
 
        We report the *minimum* of distal/posterior as the compartment gap
        (the condyle rocks between both contact regions during varus-valgus).
        """
        self.pclcond    = pclcond
        self.flex_angle = flex_angle
        self.test       = test
        savecsv = False
        plotdwells = False
        # ── Motion data ───────────────────────────────────────────────────────
        output_name = (f"{self.subject}_{self.test}_{int(compforce)}_"
                       f"{self.pclcond}_{int(self.flex_angle)}d")
        motion_output = pd.read_csv(
            self.output_dir / f"{output_name}.tab", sep='\t', skiprows=1)
        motion_output.columns = [c.strip() for c in motion_output.columns]
 
        time       = motion_output['Time'].values
        num_frames = len(time)
        
        # Index of the frame closest to t = 3.0 s
        idx_30s = int(np.argmin(np.abs(time - 3.0)))
        
        # Index of the frame closest to t = 7.5 s
        idx_75s = int(np.argmin(np.abs(time - 7.5)))
         
        # Index of the frame closest to t = 14.0 s
        idx_140s = int(np.argmin(np.abs(time - 14.0)))
         
        if verbose:
            print(f"Frames: {num_frames}")
            print(f"Frame closest to t=3.0s: index={idx_30s}, actual time={time[idx_30s]:.4f}s")
            print(f"Frame closest to t=7.5s: index={idx_75s}, actual time={time[idx_75s]:.4f}s")
            print(f"Frame closest to t=14.0s: index={idx_140s}, actual time={time[idx_140s]:.4f}s")
        
 
        euler123Insert  = []
        euler123FemComp = []
        for _, row in motion_output.iterrows():
            if self.case == 'PS':
                euler123Insert.append([
                    row['InsertMed_XFORM.X'],    row['InsertMed_XFORM.Y'],    row['InsertMed_XFORM.Z'],
                    row['InsertMed_XFORM.PSI'],  row['InsertMed_XFORM.THETA'], row['InsertMed_XFORM.PHI'],
                ])
                euler123FemComp.append([
                    row['FemComp_XFORM.X'],   row['FemComp_XFORM.Y'],   row['FemComp_XFORM.Z'],
                    row['FemComp_XFORM.PSI'], row['FemComp_XFORM.THETA'], row['FemComp_XFORM.PHI'],
                ])
            else:        
                euler123Insert.append([
                    row['Insert_XFORM.X'],    row['Insert_XFORM.Y'],    row['Insert_XFORM.Z'],
                    row['Insert_XFORM.PSI'],  row['Insert_XFORM.THETA'], row['Insert_XFORM.PHI'],
                ])
                euler123FemComp.append([
                    row['FemComp_XFORM.X'],   row['FemComp_XFORM.Y'],   row['FemComp_XFORM.Z'],
                    row['FemComp_XFORM.PSI'], row['FemComp_XFORM.THETA'], row['FemComp_XFORM.PHI'],
                ])
            
        # ── Dwell points (defined in LOCAL frame of each component) ───────────
        dwell_file   = self.data_raw_dir / "DesignDwell.xlsx"
        dwell_points = pd.read_excel(dwell_file, sheet_name=self.case)
        dwell_points = dwell_points[dwell_points['subjects'] == self.subject]
        r = dwell_points.iloc[0]
 
        # Insert plateau landmarks (local Insert frame)
        ins_med = np.array([r['insert_med_x'], r['insert_med_y'], r['insert_med_z']])
        ins_lat = np.array([r['insert_lat_x'], r['insert_lat_y'], r['insert_lat_z']])
 
        # Femoral condyle landmarks (local FemComp frame)
        fem_distmed = np.array([r['femcomp_dist_med_x'], r['femcomp_dist_med_y'], r['femcomp_dist_med_z']])
        fem_distlat = np.array([r['femcomp_dist_lat_x'], r['femcomp_dist_lat_y'], r['femcomp_dist_lat_z']])
        fem_postmed = np.array([r['femcomp_post_med_x'], r['femcomp_post_med_y'], r['femcomp_post_med_z']])
        fem_postlat = np.array([r['femcomp_post_lat_x'], r['femcomp_post_lat_y'], r['femcomp_post_lat_z']])
        
        if verbose:
            print(f"Insert medial dwell  (local): {ins_med}")
            print(f"Insert lateral dwell (local): {ins_lat}")
            print(f"FemComp dist-med     (local): {fem_distmed}")
            print(f"FemComp dist-lat     (local): {fem_distlat}")
            print(f"FemComp post-med     (local): {fem_postmed}")
            print(f"FemComp post-lat     (local): {fem_postlat}")
        
        
        # ── Per-frame point transformation ────────────────────────────────────────
        ins_med_g      = []  # global frame
        ins_lat_g      = []
        fem_distmed_g  = []
        fem_distlat_g  = []
        fem_postmed_g  = []
        fem_postlat_g  = []
        for i in range(num_frames):
            tfm_ins = euler3132tfm(euler123Insert[i])
            tfm_fem = euler3132tfm(euler123FemComp[i])
 
            ins_med_g.append(transform_point(ins_med, tfm_ins))
            ins_lat_g.append(transform_point(ins_lat, tfm_ins))
            fem_distmed_g.append(transform_point(fem_distmed, tfm_fem))
            fem_distlat_g.append(transform_point(fem_distlat, tfm_fem))
            fem_postmed_g.append(transform_point(fem_postmed, tfm_fem))
            fem_postlat_g.append(transform_point(fem_postlat, tfm_fem))
        
        
        # ── Per-frame gap computation ─────────────────────────────────────────
        records = []
        for i in range(num_frames):

            # Euclidean distances for all four fem → insert pairings
            gapmethod = 2
            if gapmethod == 1:
                distmed_vec = fem_distmed_g[i] - ins_med_g[i]
                distmed_gap = distmed_vec @ np.array([1, 0, 0])  # project onto vertical (gap) axis
                distlat_vec = fem_distlat_g[i] - ins_lat_g[i]
                distlat_gap = distlat_vec @ np.array([1, 0, 0])  # project onto vertical (gap) axis
                postmed_vec = fem_postmed_g[i] - ins_med_g[i]
                postmed_gap = postmed_vec @ np.array([1, 0, 0])  # project onto vertical (gap) axis
                postlat_vec = fem_postlat_g[i] - ins_lat_g[i]
                postlat_gap = postlat_vec @ np.array([1, 0, 0])  # project onto vertical (gap) axis
            elif gapmethod == 2:
                distmed_vec = fem_distmed_g[idx_30s] - ins_med_g[i]
                distmed_gap = distmed_vec @ np.array([1, 0, 0])  # project onto vertical (gap) axis
                distlat_vec = fem_distlat_g[idx_30s] - ins_lat_g[i]
                distlat_gap = distlat_vec @ np.array([1, 0, 0])  # project onto vertical (gap) axis
                postmed_vec = fem_postmed_g[idx_30s] - ins_med_g[i]
                postmed_gap = postmed_vec @ np.array([1, 0, 0])  # project onto vertical (gap) axis
                postlat_vec = fem_postlat_g[idx_30s] - ins_lat_g[i]
                postlat_gap = postlat_vec @ np.array([1, 0, 0])  # project onto vertical (gap) axis
            
            # gap_distmed = np.linalg.norm(fem_distmed_g - ins_med_g)
            # gap_postmed = np.linalg.norm(fem_postmed_g - ins_med_g)
            # gap_distlat = np.linalg.norm(fem_distlat_g - ins_lat_g)
            # gap_postlat = np.linalg.norm(fem_postlat_g - ins_lat_g)
 
            # Minimum gap per compartment = active (closest) contact location
            # gap_med = gap_distmed #min(gap_distmed, gap_postmed)
            # gap_lat = gap_distlat #min(gap_distlat, gap_postlat)
            
            records.append({
                'time':            time[i],
                'gap_distmed_mm':  distmed_gap,
                'gap_postmed_mm':  postmed_gap,
                'gap_distlat_mm':  distlat_gap,
                'gap_postlat_mm':  postlat_gap,
                # Global positions — used for video overlay, dropped before CSV
                '_ins_med_g':      ins_med_g[i],
                '_ins_lat_g':      ins_lat_g[i],
                '_fem_distmed_g':  fem_distmed_g[i],
                '_fem_distlat_g':  fem_distlat_g[i],
                '_fem_postmed_g':  fem_postmed_g[i],
                '_fem_postlat_g':  fem_postlat_g[i],
            })
            
        df = pd.DataFrame(records)

        out_dir  = Path(self.output_dir)
        # ── Save CSV (drop internal geometry columns) ─────────────────────────
        if savecsv:
            csv_path = out_dir / f"{output_name}_gap.csv"
            geo_cols = [c for c in df.columns if c.startswith('_')]
            df.drop(columns=geo_cols).to_csv(str(csv_path), index=False)
            if verbose:
                print(f"Gap CSV saved → {csv_path}")

        
        # ── Plot Dwell Points ──────────────────────────────────────────────────
        if plotdwells:
            frame_idx = 100  # Example: visualize the 100th frame (adjust as needed)
            meshFemComp = pv.read(self.geom_dir / "FemComp.stl")
            meshInsert  = pv.read(self.geom_dir / "Insert.stl")
            fem_t    = meshFemComp.copy().transform(euler3132tfm(euler123FemComp[frame_idx]), inplace=False)
            insert_t = meshInsert.copy().transform(euler3132tfm(euler123Insert[frame_idx]), inplace=False)
            plotter = pv.Plotter()
            add_global_axes(plotter)  # scale in mm — adjust to your geometry size
            plotter.add_mesh(fem_t,    color='lightblue', opacity=0.8, label='FemComp')
            plotter.add_mesh(insert_t, color='salmon',    opacity=0.8, label='Insert')
            for pt, color in [
                (records[frame_idx]['_ins_med_g'], 'blue'),
                (records[frame_idx]['_ins_lat_g'], 'red'),
                (records[frame_idx]['_fem_distmed_g'], 'navy'),
                (records[frame_idx]['_fem_distlat_g'], 'darkred'),
                (records[frame_idx]['_fem_postmed_g'], 'cornflowerblue'),
                (records[frame_idx]['_fem_postlat_g'], 'lightsalmon'),
            ]:
                plotter.add_mesh(pv.Sphere(radius=1.5, center=pt), color=color)
            
        
            # plotter.show()
            # plotter.close()
            # exit(1)
            
            
            # ── Plot gap vs time ──────────────────────────────────────────────────
            fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

            for ax, comp, col_post, color in [
                (axes[0], 'Medial',   'gap_postmed_mm', 'steelblue'),
                (axes[1], 'Lateral',  'gap_postlat_mm', 'tomato'),
            ]:
                # ax.plot(df['time'], df[col_dist]-df[col_dist][idx_30s], '--', color=color,
                #         alpha=0.5, linewidth=1.2, label='Distal condyle gap')
                ax.plot(df['time'], df[col_post]-df[col_post][idx_30s], '-',  color=color,
                        alpha=0.5, linewidth=1.2, label='Posterior condyle gap')
                # ax.plot(df['time'], df[col_min],  '-',  color=color,
                #         linewidth=2.0, label='Min (active) gap')
                ax.set_ylabel("Gap (mm)")
                ax.set_title(f"{comp} compartment")
                ax.legend(fontsize=8, loc='upper right')
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(f"Varus-Valgus compartment gap — {output_name}", fontsize=11)
            fig.tight_layout()
            plot_path = out_dir / f"{output_name}_gap.png"
            fig.savefig(str(plot_path), dpi=150)
            # plt.show()
            plt.close(fig)
            print(f"Gap plot saved → {plot_path}")
            
        # ── Optional: video with gap lines overlaid ───────────────────────────
        if render_video:
            meshFemComp = pv.read(self.geom_dir / "FemComp.stl")
            meshInsert  = pv.read(self.geom_dir / "Insert.stl")
 
            video_path = out_dir / f"{output_name}_gap.mp4"
            plotter    = pv.Plotter(off_screen=True, window_size=[1280, 720])
            writer     = imageio.get_writer(str(video_path), fps=30)
            print(f"Rendering gap video → {video_path}")
 
            for i, rec in enumerate(records):
                plotter.clear()
 
                fem_t    = meshFemComp.copy().transform(euler3132tfm(euler123FemComp[i]), inplace=False)
                insert_t = meshInsert.copy().transform(euler3132tfm(euler123Insert[i]), inplace=False)
 
                plotter.add_mesh(fem_t,    color='lightblue', opacity=0.7)
                plotter.add_mesh(insert_t, color='salmon',    opacity=0.7)
 
                # Gap lines: femoral dwell → insert dwell
                # Colour encodes medial (blue) vs lateral (red)
                for p0, p1, color in [
                    (rec['_fem_distmed_g'], rec['_ins_med_g'], 'navy'),
                    # (rec['_fem_postmed_g'], rec['_ins_med_g'], 'cornflowerblue'),
                    (rec['_fem_distlat_g'], rec['_ins_lat_g'], 'darkred'),
                    # (rec['_fem_postlat_g'], rec['_ins_lat_g'], 'lightsalmon'),
                ]:
                    plotter.add_mesh(pv.Line(p0, p1), color=color, line_width=3)
 
                # Dwell point spheres
                for pt, color in [
                    (rec['_ins_med_g'],     'blue'),
                    (rec['_ins_lat_g'],     'red'),
                    (rec['_fem_distmed_g'], 'navy'),
                    (rec['_fem_distlat_g'], 'darkred'),
                    # (rec['_fem_postmed_g'], 'cornflowerblue'),
                    # (rec['_fem_postlat_g'], 'lightsalmon'),
                ]:
                    plotter.add_mesh(pv.Sphere(radius=1.5, center=pt), color=color)
 
                add_global_axes(plotter, scale=30.0)
                plotter.add_axes()
                plotter.set_background('white')
                plotter.camera_position = (0, 0, 1)
                plotter.add_text(
                    f"t={rec['time']:.3f}s  |  "
                    f"Med gap = {rec['gap_distmed_mm']:.2f} mm  |  "
                    f"Lat gap = {rec['gap_distlat_mm']:.2f} mm",
                    position='upper_left', font_size=10, color='black',
                )
                writer.append_data(plotter.screenshot(return_img=True))
 
            writer.close()
            print(f"Video saved → {video_path}")
 
        # Store for downstream use
        self.gap_df = df
        medgap = df['gap_distmed_mm'][idx_75s] - df['gap_distmed_mm'][idx_30s]
        latgap = df['gap_distlat_mm'][idx_140s] - df['gap_distlat_mm'][idx_30s]

        return df, medgap, latgap
        
        
    def measureAPlaxity(self, test = 'CompPostAnt', compforce=10, pclcond='rPCL', flex_angle = 90,
                        render_video=False, verbose=False):
        """
        Quantify AP laxity during a sagittal plane drawer test.
        """
        self.pclcond    = pclcond
        self.flex_angle = flex_angle
        self.test       = test
        self.compforce   = compforce
        
        # ── Motion data ───────────────────────────────────────────────────────
        output_name = (f"{self.subject}_{self.test}_{int(self.compforce)}_"
                       f"{self.pclcond}_{int(self.flex_angle)}d")
        motion_output = pd.read_csv(
            self.output_dir / f"{output_name}.tab", sep='\t', skiprows=1)
        motion_output.columns = [c.strip() for c in motion_output.columns]
 
        time       = motion_output['Time'].values
        num_frames = len(time)
        
        # Index of the frame closest to t = 5.0 s
        idx_50s = int(np.argmin(np.abs(time - 5.0)))
        
        # Index of the frame closest to t = 7.0 s
        idx_70s = int(np.argmin(np.abs(time - 7.0)))
        
        if verbose:
            print(f"Frames: {num_frames}")
            print(f"Frame closest to t=5.0s: index={idx_50s}, actual time={time[idx_50s]:.4f}s")
            print(f"Frame closest to t=7.0s: index={idx_70s}, actual time={time[idx_70s]:.4f}s")
            
        
 
        q2 = []
        for _, row in motion_output.iterrows():
            q2.append([
                row['q2']
            ])
        q2_laxity = q2[idx_70s][0] - q2[idx_50s][0]
        
        return q2, q2_laxity
        
    
    def processFlexion(self, pclcond='rPCL', flex_angle = 90, compforce = 500):
        """ Process Passive Flexion Simulation """
        output_name = (f"{self.subject}_sim_PassiveFlexion_{compforce}_"
                       f"{pclcond}_{int(flex_angle)}d")
        motion_output = pd.read_csv(
            self.output_dir / f"{output_name}.tab", sep='\t', skiprows=1)
        motion_output.columns = [c.strip() for c in motion_output.columns]

        
        print(f"Processing flexion data for {output_name}...")
        print(motion_output.head())