import numpy as np
import pandas as pd
from registrations import read_tfm_file, tfm2euler123, write_tfm_file
from pathlib import Path


class createElementsMixin:
    def create_elements(self, verbose=False, run_adams=True):
        cmd_file = self.cmd_path("step2_create_elements")
        if verbose:
            print(f"Creating simulation CMD file at: {cmd_file}")
        with open(cmd_file, "w", encoding="utf-8") as fid:
            self._write_create_elements(fid, verbose=verbose)
        if run_adams:
            self.run_adams(cmd_file)
    
    
    def _read_lig_measurements(self, filepath):
        """
        Reads ligament measurement data from the given file.
        """
        data = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # --- Find the line that contains "3D Distance"
        start_idx = None
        for i, line in enumerate(lines):
            if "3D Distance" in line:
                start_idx = i
                break

        if start_idx is None:
            raise ValueError("'3D Distance' section not found in file.")

        # Skip the next two lines after "3D Distance"
        idx = start_idx + 2

        # --- Parse ligament measurement rows
        while idx < len(lines):
            line = lines[idx].strip()

            # Stop if line has no alphabetic characters (behavior of MATLAB version)
            if not any(c.isalpha() for c in line):
                break

            # Split by whitespace (MATLAB: strsplit with ' ', '\t')
            parts = line.split()

            # First token is ligament name
            name = parts[0]

            # Remaining numeric fields: MATLAB looked for items containing '.'
            numeric_values = []
            for token in parts[1:]:
                if "." in token:  
                    try:
                        numeric_values.append(float(token))
                    except ValueError:
                        pass  # ignore anything weird

            data.append([name] + numeric_values)
            idx += 1

        return data
    
    
    
    
    def _write_ligaments_file(self, model_inputs_dir, input_ligs='Ligament_Measurements.txt', verbose=False):
        """
        Automatically populate the ligaments xlsx file from the text file
        exported from Mimics. Equivalent to writeLigamentsFile.m.
    
        Parameters
        ----------
        model_inputs_dir : Path or str
            Path to the model_inputs directory for the subject/case.
        input_ligs : str
            Filename of the ligament measurements text file.
        """
        # ── Load input measurement file ───────────────────────────────────────────
        arg = model_inputs_dir / input_ligs
        data = self._read_lig_measurements(arg)
        # `data` is expected to be a list-of-lists where:
        #   col 0  : ligament name
        #   cols 1-3: FemX, FemY, FemZ  (1-based cols 2-4 in MATLAB)
        #   cols 4-6: TibX, TibY, TibZ  (1-based cols 5-7 in MATLAB)
        #   col 7  : Length             (1-based col 8 in MATLAB)
    
        # ── Load source xlsx files ────────────────────────────────────────────────
        file_df = pd.read_excel(model_inputs_dir / "Ligaments_source.xlsx")
        smcl_df = pd.read_excel(model_inputs_dir / "sMCLAttachments_source.xlsx", header=None)
    
        # Convert to list-of-lists for index-based manipulation (mirrors MATLAB cell array)
        file_cells = file_df.values.tolist()
        smcl_cells = smcl_df.values.tolist()
        # print(type(smcl_cells))
        
        # ── Rewrite Ligaments File ────────────────────────────────────────────────
        file_names = [str(row[0]).strip() for row in file_cells]
        # print("Original ligament attachment names in Excel:", file_names)
        for data_row in data:
            lig_name = str(data_row[0]).replace(' ', '')
    
            if lig_name not in file_names:
                continue
    
            row_idx = file_names.index(lig_name)
    
            # z-value direction check: FemZ is col 3 (0-based), TibZ is col 6
            # Femur z should be greater — if not, swap origin and insertion
            try:
                fem_z = float(data_row[3])
                tib_z = float(data_row[6])
            except (ValueError, TypeError):
                fem_z = float(str(data_row[3]))
                tib_z = float(str(data_row[6]))
    
            if fem_z < tib_z:
                # Swap: put tibia coords in femur columns and vice versa
                
                file_cells[row_idx][1:8] = [
                    data_row[4], data_row[5], data_row[6],  # TibX, TibY, TibZ
                    data_row[1], data_row[2], data_row[3],  # FemX, FemY, FemZ
                    data_row[7],                             # Length
                ]
            else:
                # Already in correct order (femur first)
                
                file_cells[row_idx][1:8] = data_row[1:8]
    
        # ── Rewrite sMCL File ─────────────────────────────────────────────────────
        # Each tuple: (target name in smcl, source name in data, 0-based col slice)
        smcl_pairs = [
            ('sMCL_A_Fem',    'sMCL_Prox_A', slice(1, 4)),  
            ('sMCL_A_Sphere', 'sMCL_Prox_A', slice(4, 7)),  
            ('sMCL_A_Tib',    'sMCL_Dist_A', slice(4, 7)),
            ('sMCL_C_Fem',    'sMCL_Prox_C', slice(1, 4)),
            ('sMCL_C_Sphere', 'sMCL_Prox_C', slice(4, 7)),
            ('sMCL_C_Tib',    'sMCL_Dist_C', slice(4, 7)),
            ('sMCL_P_Fem',    'sMCL_Prox_P', slice(1, 4)),
            ('sMCL_P_Sphere', 'sMCL_Prox_P', slice(4, 7)),
            ('sMCL_P_Tib',    'sMCL_Dist_P', slice(4, 7)),
        ]

        
        smcl_names = [str(row[0]).strip() for row in smcl_cells]        
        data_names = [str(row[0]).strip() for row in data] 
        
        for smcl_target, data_source, col_slice in smcl_pairs:            
            if smcl_target not in smcl_names:
                print(f"Warning: sMCL target '{smcl_target}' not found in sMCL Excel file. Skipping.")
                continue
            smcl_idx = smcl_names.index(smcl_target)
            if data_source not in data_names:
                print(f"Warning: data source '{data_source}' not found in ligament measurements. Skipping.")
                continue
            data_idx = data_names.index(data_source)
            smcl_cells[smcl_idx][1:4] = data[data_idx][col_slice]
        
        # ── Reconstruct DataFrames with headers ───────────────────────────────────
        file_headers = [
            'LigamentFiber', 'FemX', 'FemY', 'FemZ',
            'TibX', 'TibY', 'TibZ', 'Length',
            'Origin', 'Insertion', 'MainLoMultiplier',
            'SecondaryL0Multiplier', 'Fibers',
            'LinearStiffness', 'ToeRegionCharacteristic',
        ]
        smcl_headers = ['sMCL_marks', 'X', 'Y', 'Z']
        file_out = pd.DataFrame(file_cells, columns=file_headers)
        smcl_out = pd.DataFrame(smcl_cells, columns=smcl_headers)
        if verbose:
            print("Updated ligament attachment data based on measurements:")
            print(file_out.head())
            print(smcl_out.head())
        
        # ── Save files ────────────────────────────────────────────────────────────
        file_out.to_excel(model_inputs_dir / "Ligament_Attachments_CT.xlsx", index=False)
        smcl_out.to_excel(model_inputs_dir / "sMCL_Attachments_CT.xlsx",    index=False)
    
    
    


    def ligament_transformation(self, frameone, frametwo, verbose=False):
        def apply_transform(points, R):
            """Apply a rotation + translation to Nx3 points."""
            if verbose:
                print("Applying transformation with R:\n", R)
                print("Original points:\n", points)
            return (R @ points.T).T
        # -----------------------------
        # Build directories
        # -----------------------------
        study =              self.study_root
        model_inputs_dir =  self.model_inputs_dir
        tfm_dir =           self.transforms_dir

        # -----------------------------
        # Read transformation matrices
        # -----------------------------
        R_Fem, _ = read_tfm_file(tfm_dir / f"Fem{frameone}2{frametwo}.tfm")
        R_Tib, _ = read_tfm_file(tfm_dir / f"Tib{frameone}2{frametwo}.tfm")
        
        if verbose:
            print("R_Fem:\n", R_Fem)
            print("R_Tib:\n", R_Tib)
        
        # Convert translations to numpy arrays
        # T_Fem = np.asarray(T_Fem).reshape(3)
        # T_Tib = np.asarray(T_Tib).reshape(3)

        # -----------------------------
        # Read ligament attachments
        # -----------------------------
        lig_file_1 = model_inputs_dir / f"Ligament_Attachments_{frameone}.xlsx"
        lig_file_2 = model_inputs_dir / f"Ligament_Attachments_{frametwo}.xlsx"

        lig_df = pd.read_excel(lig_file_1).copy()

        # -----------------------------
        # FEM & TIB points (vectorized)
        # -----------------------------
        fem_cols = ["FemX", "FemY", "FemZ"]
        tib_cols = ["TibX", "TibY", "TibZ"]

        fem_points = lig_df[fem_cols].to_numpy()
        tib_points = lig_df[tib_cols].to_numpy()
        fem_points_aug = np.hstack([fem_points, np.ones((fem_points.shape[0], 1))])  # Nx4 homogeneous
        tib_points_aug = np.hstack([tib_points, np.ones((tib_points.shape[0], 1))])  # Nx4 homogeneous
        fem_points_p = apply_transform(fem_points_aug, R_Fem)
        tib_points_p = apply_transform(tib_points_aug, R_Tib)
        
        lig_df[fem_cols] = fem_points_p[:, :3]  # drop homogeneous coordinate
        lig_df[tib_cols] = tib_points_p[:, :3]  # drop homogeneous coordinate
        
        
        lig_df.to_excel(lig_file_2, index=False)

        # -----------------------------
        # sMCL sheet
        # -----------------------------
        smcl_file_1 = model_inputs_dir / f"sMCL_Attachments_{frameone}.xlsx"
        smcl_file_2 = model_inputs_dir / f"sMCL_Attachments_{frametwo}.xlsx"

        smcl_df = pd.read_excel(smcl_file_1).copy()

        # Which marks belong to which bone?
        fem_marks = ["sMCL_A_Fem", "sMCL_C_Fem", "sMCL_P_Fem"]
        tib_marks = ["sMCL_A_Sphere", "sMCL_C_Sphere", "sMCL_P_Sphere",
                    "sMCL_A_Tib", "sMCL_C_Tib", "sMCL_P_Tib"]

        # -----------------------------
        # Helper: transform subgroup
        # -----------------------------
        def transform_marks(df, marks, R):
            mask = df["sMCL_marks"].isin(marks)
            pts = df.loc[mask, ["X", "Y", "Z"]].to_numpy()
            pts_aug = np.hstack([pts, np.ones((pts.shape[0], 1))])  # Nx4 homogeneous
            pts_p = apply_transform(pts_aug, R)[:, :3]  # drop homogeneous coordinate
            df.loc[mask, ["X", "Y", "Z"]] = pts_p

        # -----------------------------
        # Apply transform per bone group
        # -----------------------------
        transform_marks(smcl_df, fem_marks, R_Fem)
        transform_marks(smcl_df, tib_marks, R_Tib)

        smcl_df.to_excel(smcl_file_2, index=False)







    def createTwrtF(self):
        # -----------------------------
        # Paths
        # -----------------------------
        study   = self.study_root
        tfm_dir = self.transforms_dir
        
        method = 1  # 1 = CT → FC and CT → TC; 2 = TC → FC only
        # Output file
        # output_file = tfm_dir / "TwrtF.tfm"

        # -----------------------------
        # Method 1: CT → FC and CT → TC
        # -----------------------------
        if method == 1:
            R_CT2FC, _ = read_tfm_file(tfm_dir / "TibCT2FC.tfm")
            R_CT2TC, _= read_tfm_file(tfm_dir / "TibCT2TC.tfm")

            TwrtF = R_CT2FC @ np.linalg.inv(R_CT2TC)

        # -----------------------------
        # Method 2: TC → FC only
        # -----------------------------
        elif method == 2:
            R_TC2FC, _ = read_tfm_file(tfm_dir / "TC2FC.tfm")
            TwrtF = np.linalg.inv(R_TC2FC)

        else:
            raise ValueError("Method must be 1 or 2.")

        # -----------------------------
        # Write output transform
        # -----------------------------
        write_tfm_file(TwrtF, tfm_dir, "TwrtF.tfm")

        return TwrtF


    def _write_create_elements(self, fid, verbose=False):
        
        def _strip_df(df):
            return df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
        
        self._write_ligaments_file(self.model_inputs_dir, verbose=False)
        self.ligament_transformation('CT', 'FC')
        self.createTwrtF()
        

        model_name = f"{self.subject}"            
        origin_xform,    _      =      read_tfm_file( self.transforms_dir / "RI.tfm")
        insertion_xform, _      =      read_tfm_file( self.transforms_dir / "RI.tfm")
        TwrtF, euler_xyz_TwrtF  =    read_tfm_file( self.transforms_dir / "TwrtF.tfm")
        
        
        # ── Input files ──────────────────────────────────────────────────────
        ligaments_data          = _strip_df(pd.read_excel(self.model_inputs_dir / "Ligament_Attachments_FC.xlsx"))
        design_variables_data   = _strip_df(pd.read_excel(self.model_inputs_dir / "Design Variables.xlsx"))
        sensor_data             = _strip_df(pd.read_excel(self.model_inputs_dir / "Sensors.xlsx"))
        constraint_data         = _strip_df(pd.read_excel(self.model_inputs_dir / "Constraints.xlsx"))
        measure_data            = _strip_df(pd.read_excel(self.model_inputs_dir / "Measures.xlsx"))
        correction_data         = _strip_df(pd.read_excel(self.model_inputs_dir / "DisplacementCorrection.xlsx"))
        spline_data             = _strip_df(pd.read_excel(self.model_inputs_dir / "Spline_Characteristics.xlsx"))
        
        
        # ── Tibia wrt Femur markers ───────────────────────────────────────────
        # AP marker
        ap_z_unit = np.cross([0, 1, 0], TwrtF[:3, 0])
        ap_x_unit = np.cross(ap_z_unit, [0, 1, 0])
        ap_y_unit = np.cross(ap_z_unit, ap_x_unit)
        ap_marker_xform = np.array([
            [ap_x_unit[0], ap_y_unit[0], ap_z_unit[0], 0],
            [ap_x_unit[1], ap_y_unit[1], ap_z_unit[1], 0],
            [ap_x_unit[2], ap_y_unit[2], ap_z_unit[2], 0],
            [0,            0,            0,            1],
        ])
        ap_marker_xyz123 = tfm2euler123(ap_marker_xform)
 
        # PD marker
        pd_marker_xform = TwrtF @ np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
        pd_marker_xyz123 = tfm2euler123(pd_marker_xform)
        
        # ── Secondary L0 multiplier flags ────────────────────────────────────
        # for row in ligaments_data['SecondaryL0Multiplier']:
        #     if str(row).strip() == 'nan':
        #         print("Found NaN in SecondaryL0Multiplier column. Setting to 0.")
        #     else:
        #         print(f"SecondaryL0Multiplier value: {row}")
        secondary_multiplier_flag = [
            0 if str(row).strip() == 'nan' else 1
            for row in ligaments_data['SecondaryL0Multiplier']
        ]
        
        # ── Read Binary File  ───────────────────────────────────────
        input_bin_file = f'{self.subject}_C1'
        fid.write('! ----- Binary File ----- !\n!\n')
        fid.write(f'file bin read file="{self.bin_dir}/{input_bin_file}.bin" \n!\n')
        
        
        
        # ── Chapter 1: Design Variables ───────────────────────────────────────
        fid.write('! ----- Design Variables ----- !\n!\n')
        for _, row in design_variables_data.iterrows():
            variable_name = str(row['Name'])
            dv_value      = row['Value']
            fid.write(f'variable create  variable_name = .{model_name}.{variable_name} &\n')
            fid.write('    units = "no_units" &\n')
            fid.write('    range = -1.0, 1.0 &\n')
            fid.write('    use_allowed_values = no &\n')
            fid.write('    delta_type = relative &\n')
            fid.write(f'    real_value = {dv_value:f}\n')
            fid.write('!\n')
 
        # Toe region design variables
        dv_prefixes     = ['DV_A_', 'DV_B_', 'transitionX_', 'transitionY_']
        spline_headers  = list(spline_data.columns)
        for _, s_row in spline_data.iterrows():
            for j, prefix in enumerate(dv_prefixes):
                var_name = prefix + str(s_row['Lig Name'])
                fid.write(f'variable create  variable_name = .{model_name}.{var_name} &\n')
                fid.write('    units = "no_units" &\n')
                fid.write('    range = -1.0, 1.0 &\n')
                fid.write('    use_allowed_values = no &\n')
                fid.write('    delta_type = relative &\n')
                fid.write(f'    real_value = {s_row[spline_headers[j + 1]]:f}\n')
                fid.write('!\n')
                
        
        # ── Chapter 2: Ligaments ──────────────────────────────────────────────
        fid.write('! ----- Ligament Markers ----- !\n!\n')
        for _, row in ligaments_data.iterrows():
            ligament       = str(row['LigamentFiber'])
            origin_part    = str(row['Origin'])
            insertion_part = str(row['Insertion'])
 
            origin_imaging    = np.array([row['FemX'], row['FemY'], row['FemZ']])
            insertion_imaging = np.array([row['TibX'], row['TibY'], row['TibZ']])
 
            origin_build    = origin_xform    @ np.append(origin_imaging, 1)
            insertion_build = insertion_xform @ np.append(insertion_imaging, 1)
 
            # Origin marker
            fid.write(f'marker create  marker_name =.{model_name}.{origin_part}.{origin_part}_{ligament} &\n')
            fid.write(f'    location = {origin_build[0]:f},{origin_build[1]:f},{origin_build[2]:f} &\n')
            fid.write('    orientation = 0.0d, 0.0d, 0.0d\n')
            fid.write(f'marker attributes  marker_name ={origin_part}_{ligament} &\n')
            fid.write('    visibility = off\n')
            fid.write('!\n')
 
            # Insertion marker
            fid.write(f'marker create  marker_name =.{model_name}.{insertion_part}.{insertion_part}_{ligament} &\n')
            fid.write(f'    location = {insertion_build[0]:f},{insertion_build[1]:f},{insertion_build[2]:f} &\n')
            fid.write('    orientation = 0.0d, 0.0d, 0.0d\n')
            fid.write(f'marker attributes  marker_name ={insertion_part}_{ligament} &\n')
            fid.write('    visibility = off\n')
            fid.write('!\n')
            fid.write('!\n')
            
        # ── Ligament Forces ───────────────────────────────────────────────────
        fid.write('! ----- Ligament Forces ----- !\n!\n')
        for _, row in ligaments_data.iterrows():
            ligament       = str(row['LigamentFiber'])
            origin_part    = str(row['Origin'])
            insertion_part = str(row['Insertion'])
            fid.write(f'force create direct single_component_force  single_component_force_name ={ligament} &\n')
            fid.write('    type_of_freedom = translational &\n')
            fid.write(f'    i_marker_name = {origin_part}_{ligament} &\n')
            fid.write(f'    j_marker_name = {insertion_part}_{ligament} &\n')
            fid.write('    action_only = off &\n')
            fid.write('    function = ""\n')
            fid.write(f'force attributes  force_name ={ligament} &\n')
            fid.write('    visibility = on &\n')
            fid.write('    size_of_icons = 10.0 &\n')
            fid.write('    name_visibility = off\n')
            fid.write('!\n')
            
            
        # ── L0 ────────────────────────────────────────────────────────────────
        fid.write('! ----- L0 ----- !\n!\n')
        for i, row in ligaments_data.iterrows():
            ligament         = str(row['LigamentFiber'])
            main_multiplier  = str(row['MainLoMultiplier'])
            l0_value         = row['Length']
            fid.write(f'variable create  variable_name = L0_{ligament} &\n')
            fid.write('    units = "no_units" &\n')
            fid.write('    range = -1.0, 1.0 &\n')
            fid.write('    use_allowed_values = no &\n')
            fid.write('    delta_type = relative &\n')
            if secondary_multiplier_flag[i] == 0:
                fid.write(f'    real_value = ( Percent_L0_{main_multiplier} * {l0_value:f} )\n')
            else:
                secondary_multiplier = str(row['SecondaryL0Multiplier'])
                fid.write(f'    real_value = ( Percent_L0_{secondary_multiplier} * Percent_L0_{main_multiplier} * {l0_value:f} )\n')
            fid.write('!\n')
 
        # ── Force Measures ────────────────────────────────────────────────────
        fid.write('! ----- Force Measures ----- !\n!\n')
        for _, row in ligaments_data.iterrows():
            ligament = str(row['LigamentFiber'])
            fid.write(f'measure create object  measure_name = .{model_name}.Force_{ligament} &\n')
            fid.write('    from_first = yes &\n')
            fid.write(f'    object = .{model_name}.{ligament} &\n')
            fid.write('    characteristic = element_force &\n')
            fid.write('    component = mag_component &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
 
        # ── Length Measures ───────────────────────────────────────────────────
        fid.write('! ----- Length Measures ----- !\n!\n')
        for _, row in ligaments_data.iterrows():
            ligament       = str(row['LigamentFiber'])
            origin_part    = str(row['Origin'])
            insertion_part = str(row['Insertion'])
            # print(f"Creating length measure for ligament {ligament} between {origin_part} and {insertion_part}")
            fid.write(f'measure create function  measure_name =.{model_name}.Length_{ligament} &\n')
            fid.write(f'    function = "DM({origin_part}_{ligament},{insertion_part}_{ligament})" &\n')
            fid.write('    units = length &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
            
        # ── VR Measures ───────────────────────────────────────────────────────
        fid.write('! ----- VR Measures ----- !\n!\n')
        for _, row in ligaments_data.iterrows():
            ligament       = str(row['LigamentFiber'])
            origin_part    = str(row['Origin'])
            insertion_part = str(row['Insertion'])
            fid.write(f'measure create function  measure_name =VR_{ligament} &\n')
            fid.write(f'    function = "VR({origin_part}_{ligament},{insertion_part}_{ligament})" &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
 
        # ── StepVR Measures ───────────────────────────────────────────────────
        fid.write('! ----- StepVR Measures ----- !\n!\n')
        for _, row in ligaments_data.iterrows():
            ligament = str(row['LigamentFiber'])
            fid.write(f'measure create function  measure_name =StepVR_{ligament} &\n')
            fid.write(f'    function = "VR_{ligament}*Step(VR_{ligament}, 0, 0, VR_{ligament} +0.1,1)" &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
 
        # ── Displacement Measures ─────────────────────────────────────────────
        fid.write('! ----- Displacement Measures ----- !\n!\n')
        for _, row in ligaments_data.iterrows():
            ligament = str(row['LigamentFiber'])
            fid.write(f'measure create function  measure_name = Disp_{ligament} &\n')
            fid.write(f'    function = "(Length_{ligament}-L0_{ligament})" &\n')
            fid.write('    units = length &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
 
        # ── Strain Measures ───────────────────────────────────────────────────
        fid.write('! ----- Strain Measures ----- !\n!\n')
        for _, row in ligaments_data.iterrows():
            ligament = str(row['LigamentFiber'])
            fid.write(f'measure create function  measure_name = Strain_{ligament} &\n')
            fid.write(f'    function = "(Disp_{ligament}/L0_{ligament})" &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
        
        
        # ── Chapter 3: Additional Markers (Joints and Forces) ─────────────────
    
        joint_markers = [
            'Ground.Ground_MomVV_0d', 'Ground.FixFemToGround', 'Ground.FixTibToGround',        # 1
            'Ground.Ground_ForceAP_0d', 'Tib.FixTrayToTib', 'Insert.FixInsertToTray', 'Fib.FixFibToTib',  # 2
            'Tray.FixTrayToTib', 'Tray.FixInsertToTray', 'Tib.FixFibToTib',                   # 3
            'FemComp.FixFemCompToFem', 'Fem.FixFemToGround',                                   # 4
            'Fem.FixFemCompToFem', 'Fem.Axial_Constrain', 'Ground.FixFlex_0d',                 # 5
            'Fem.Global_Origin', 'Fem.Fem_APAxis_0d',                                          # 6
            'Tib.FixTibToGround', 'Tib.Axial_Constraint', 'Fem.Axial_Constraint',             # 7
            'Tib.FixFlex_0d', 'Tib.Tib_FlexionAxis_0d', 'Tib.TibRef',                        # 8
            'Tib.TibRef_AtFemOrigin', 'Tib.Tib_MomVV_0d', 'Tib.Tib_ForceAP_0d',             # 9
            'Tib.Tib_MomIE', 'Tib.Tib_ForceCD', 'Tib.Tib_ForceCD_2', 'Tib.Tib_MomIE_2',    #10
            'Ground.Ground_FemFlexionJoint', 'Fem.Fem_FemFlexionJoint',   #11
        ]
        locations = (
            [[0,0,0]]*3  #1
            + [[0,0,0]]*4 #2
            + [[0,0,0]]*3 #3
            + [[0,0,0]]*2 #4
            + [[0,0,0]]*3 #5
            + [[0,0,0]]*2 #6
            + [[0,0,0]]*3 #7
            + [[0,0,0]]*2 + [euler_xyz_TwrtF[:3].tolist()] #8
            + [[0,0,0]]*3 #9
            + [[0,0,0]]*4 #10
            + [[0,0,0]]*2 #11
        )
        orientations = (
            [euler_xyz_TwrtF[3:6].tolist(), [0,0,0], [0,0,0]]                           # 1
            + [ap_marker_xyz123[3:6].tolist(), [0,0,0], [0,0,0], [0,0,0]]             # 2
            + [[0,0,0]]*3                                                               # 3
            + [[0,0,0]]*2                                                               # 4
            + [[0,0,0], [0,0,0], [90,90,0]]                                            # 5
            + [[0,0,0], ap_marker_xyz123[3:6].tolist()]                                # 6
            + [[0,0,0], pd_marker_xyz123[3:6].tolist(), pd_marker_xyz123[3:6].tolist()] # 7
            + [[0,0,0], [0,0,0], euler_xyz_TwrtF[3:6].tolist()]                         # 8
            + [euler_xyz_TwrtF[3:6].tolist(), ap_marker_xyz123[3:6].tolist(), ap_marker_xyz123[3:6].tolist()]  # 9
            + [pd_marker_xyz123[3:6].tolist()]*4
            + [[90,0,0], [90,0,0]]
        )
            
        fid.write('! ----- Joint Markers ----- !\n!\n')
        for marker, loc, ori in zip(joint_markers, locations, orientations):
            fid.write(f'marker create  marker_name =.{model_name}.{marker} &\n')
            fid.write(f'    location = {loc[0]:f},{loc[1]:f},{loc[2]:f} &\n')
            fid.write(f'    orientation = {ori[0]:f}d,{ori[1]:f}d,{ori[2]:f}d\n')
            fid.write(f'marker attributes  marker_name =.{model_name}.{marker} &\n')
            fid.write('    visibility = off\n\n')
            fid.write('!\n')
            
        
        # ── Chapter 4: Tibiofemoral Contacts ──────────────────────────────────
        contact_names = ['FCICont']
        i_geometries  = ['FemComp.SOLID2']
        j_geometries  = ['Insert.SOLID3']
            
 
        fid.write('! ----- Contact Forces ----- !\n!\n')
        for cname, i_geom, j_geom in zip(contact_names, i_geometries, j_geometries):
            fid.write(f'contact create  contact_name = .{model_name}.{cname} &\n')
            fid.write(f'    i_geometry_name = .{model_name}.{i_geom} &\n')
            fid.write(f'    j_geometry_name = .{model_name}.{j_geom} &\n')
            fid.write(f'    stiffness = (.{model_name}.FCIContStiffness) &\n')
            fid.write(f'    damping = (.{model_name}.FCIContDamping) &\n')
            fid.write(f'    exponent = (.{model_name}.FCIContForceExp) &\n')
            fid.write(f'    dmax = (.{model_name}.FCIContPenetrationDepth)\n')
            fid.write('!\n')
            fid.write(f'force attributes  force_name = .{model_name}.{cname} &\n')
            fid.write('    visibility = off\n')
            fid.write('!\n')
 
        fid.write('! ----- Contact Graphics ----- !\n!\n')
        for cname in contact_names:
            fid.write(f'geometry create shape gcontact  contact_force_name = .{model_name}.Graphic_{cname} &\n')
            fid.write(f'    contact_element_name = .{model_name}.{cname} &\n')
            fid.write('    force_display = components\n')
            fid.write(f'geometry attributes  geometry_name = .{model_name}.Graphic_{cname} &\n')
            fid.write('    color = YellowGreen\n')
            fid.write('!\n')
 
        # ── Chapter 5: Joints and Motion Elements ─────────────────────────────
        fixed_joints = ['FixFemCompToFem', 'FixFemToGround', 'FixInsertToTray', 'FixTrayToTib', 'FixFibToTib', 'FixTibToGround']
        i_markers    = ['FemComp.FixFemCompToFem', 'Fem.FixFemToGround', 'Tray.FixInsertToTray', 'Tray.FixTrayToTib', 'Fib.FixFibToTib', 'Tib.FixTibToGround']
        j_markers    = ['Fem.FixFemCompToFem', 'ground.FixFemToGround', 'Insert.FixInsertToTray', 'Tib.FixTrayToTib', 'Tib.FixFibToTib', 'ground.FixTibToGround']

        
        fid.write('! ----- Fixed Joints ----- !\n!\n')
        for joint, i_m, j_m in zip(fixed_joints, i_markers, j_markers):
            fid.write(f'constraint create joint fixed  joint_name =.{model_name}.{joint} &\n')
            fid.write(f'    i_marker =.{model_name}.{i_m} &\n')
            fid.write(f'    j_marker =.{model_name}.{j_m}\n')
            fid.write('!\n')
            fid.write(f'constraint attributes  constraint_name =.{model_name}.{joint} &\n')
            fid.write('    visibility = off\n')
            fid.write('!\n')
 
        fid.write('! ----- Axial Joint ----- !\n!\n')
        fid.write(f'constraint create joint translational  joint_name =.{model_name}.AxialConstraint &\n')
        fid.write(f'    i_marker =.{model_name}.Tib.Axial_Constraint &\n')
        fid.write(f'    j_marker =.{model_name}.Fem.Axial_Constraint\n')
        fid.write(f'constraint attributes  constraint_name =.{model_name}.AxialConstraint &\n')
        fid.write('    visibility = off\n')
        fid.write('!\n')
 
        fid.write('! ----- Axial Motion ----- !\n!\n')
        fid.write(f'constraint create motion motion_name =.{model_name}.AxialTibTranslation &\n')
        fid.write(f'    joint=.{model_name}.AxialConstraint &\n')
        fid.write('    type=translational &\n')
        fid.write('    time_derivative=displacement &\n')
        fid.write('    function="-2*step(time,0,0,.5,1)"\n')
        fid.write(f'constraint attributes  constraint_name =.{model_name}.AxialTibTranslation &\n')
        fid.write('    visibility = off\n')
        fid.write('!\n')
 
        fid.write('! ----- Femur Flexion Joint ----- !\n!\n')
        fid.write(f'constraint create joint Revolute  joint_name =.{model_name}.FemFlexion &\n')
        fid.write(f'    i_marker =.{model_name}.Ground.Ground_FemFlexionJoint &\n')
        fid.write(f'    j_marker =.{model_name}.Fem.Fem_FemFlexionJoint\n')
        fid.write(f'constraint attributes  constraint_name =.{model_name}.FemFlexion &\n')
        fid.write('    visibility = off\n')
        fid.write('!\n')
 
        fid.write('! ----- Flexion Motion ----- !\n!\n')
        fid.write(f'constraint create motion motion_name =.{model_name}.FemFlexionRotation &\n')
        fid.write(f'    joint=.{model_name}.FemFlexion &\n')
        fid.write('    type=rotational &\n')
        fid.write('    time_derivative=displacement &\n')
        fid.write('    function="Step(time, 1, 0, 91, 90*DTOR)"\n')
        fid.write(f'constraint attributes  constraint_name =.{model_name}.FemFlexionRotation &\n')
        fid.write('    visibility = off\n')
        fid.write('!\n')
 
        fid.write('! ----- Primitive Joint ----- !\n!\n')
        fid.write(f'constraint create primitive_joint Perpendicular  jprim_name =.{model_name}.FixFlex_0d &\n')
        fid.write(f'    i_marker =.{model_name}.Tib.FixFlex_0d &\n')
        fid.write(f'    j_marker =.{model_name}.Ground.FixFlex_0d\n')
        fid.write('constraint attributes &\n')
        fid.write(f'    constraint_name =.{model_name}.FixFlex_0d &\n')
        fid.write('    visibility = off\n')
        fid.write('!\n')

        
        
        
        
        
        
        
        # 555555555555555555555555555555555555555555555555555555555555555555555555555
        # ── Chapter 6: External F/Ts ──────────────────────────────────────────
        force_torques      = ['ForceCD', 'ForceAP_0d', 'Mom_IE', 'MomVV_0d']
        freedom            = ['translational', 'translational', 'rotational', 'rotational']
        ft_i_markers       = ['Tib.Tib_ForceCD', 'Tib.Tib_ForceAP_0d', 'Tib.Tib_MomIE', 'Tib.Tib_MomVV_0d']
        ft_j_markers       = ['Tib.Tib_ForceCD_2', 'ground.Ground_ForceAP_0d', 'Tib.Tib_MomIE_2', 'ground.Ground_MomVV_0d']
        function_strings   = [
            '10*step(time,0,0,.5,1)',
            '134*step(time,1,0,3,1)',
            '4000*step(time,2.5,0,4.5,1)',
            '8000*step(time,.5,0,2.5,1)',
        ]
 
        fid.write('! ----- External F/T ----- !\n!\n')
        for ft, fr, i_m, j_m, fn in zip(force_torques, freedom, ft_i_markers, ft_j_markers, function_strings):
            fid.write(f'force create direct single_component_force  single_component_force_name = .{model_name}.{ft} &\n')
            fid.write(f'    type_of_freedom = {fr} &\n')
            fid.write(f'    i_marker_name = .{model_name}.{i_m} &\n')
            fid.write(f'    j_marker_name = .{model_name}.{j_m} &\n')
            fid.write('    action_only = on &\n')
            fid.write(f'    function = "{fn}"\n')
            fid.write('!\n')
            fid.write('force attributes &\n')
            fid.write(f'    force_name = .{model_name}.{ft} &\n')
            fid.write('    size_of_icons = 25.0\n')
            fid.write('!\n')
 
        graphics = [
            'ForceCD_force_graphic_1', 'ForceAP_0d_force_graphic_1',
            'Mom_IE_force_graphic_1', 'MomVV_0d_force_graphic_1',
        ]
 
        fid.write('! ----- External F/T Graphics ----- !\n!\n')
        for graphic, ft, i_m in zip(graphics, force_torques, ft_i_markers):
            fid.write('geometry create shape force &\n')
            fid.write(f'    force_name = .{model_name}.{graphic} &\n')
            fid.write(f'    force_element_name = .{model_name}.{ft} &\n')
            fid.write(f'     applied_at_marker_name = .{model_name}.{i_m}\n')
            fid.write('geometry attributes &\n')
            fid.write(f'    geometry_name = .{model_name}.{graphic} &\n')
            fid.write('    active = off\n')
            fid.write('!\n')
 
        # ── Chapter 7: Measures ───────────────────────────────────────────────
        measures         = ['Applied_ForceCD', 'Applied_Mom_IE', 'Applied_ForceAP_0d', 'Applied_Mom_VV_0d']
        objects          = ['ForceCD', 'Mom_IE', 'ForceAP_0d', 'MomVV_0d']
        characteristics  = ['element_force', 'element_torque', 'element_force', 'element_torque']
 
        fid.write('! ----- External F/T Measures ----- !\n!\n')
        for meas, obj, char in zip(measures, objects, characteristics):
            fid.write(f'measure create object  measure_name = .{model_name}.{meas} &\n')
            fid.write('    from_first = yes &\n')
            fid.write(f'    object = .{model_name}.{obj} &\n')
            fid.write(f'    characteristic = {char} &\n')
            fid.write('    component = mag_component &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
            fid.write(f'data_element attributes  data_element_name = .{model_name}.{meas} &\n')
            fid.write('    active = off &\n')
            fid.write('    color = WHITE\n')
            fid.write('!\n')
 
        # Translation measures
        trans_measures  = ['ZwrtFem', 'YwrtFem', 'XwrtFem']
        components      = ['z', 'y', 'x']
 
        fid.write('! ----- Translation Measures ----- !\n!\n')
        for meas, comp in zip(trans_measures, components):
            fid.write(f'measure create pt2pt  measure_name = .{model_name}.{meas} &\n')
            fid.write(f'    from_point = .{model_name}.Fem.Global_Origin &\n')
            fid.write(f'    to_point = .{model_name}.Tib.TibRef_AtFemOrigin &\n')
            fid.write('    characteristic = translational_displacement &\n')
            fid.write(f'    component = {comp}_componen &\n')
            fid.write(f'    coordinate_rframe = .{model_name}.Fem.Global_Origin &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
            fid.write(f'data_element attributes  data_element_name = .{model_name}.{meas} &\n')
            fid.write('    color = WHITE\n')
            fid.write('!\n')
 
        # TwrtF measures
        fid.write('! ----- TwrtF Measures ----- !\n!\n')
        for i in range(1, 4):
            for j in range(1, 4):
                measure_name = f'TwrtF{i}{j}'
                component    = f'mat_{i}_{j}_component'
                fid.write(f'measure create orient  measure_name = .{model_name}.{measure_name} &\n')
                fid.write(f'    to_frame = .{model_name}.Tib.TibRef &\n')
                fid.write(f'    from_frame = .{model_name}.Fem.Global_Origin &\n')
                fid.write('    characteristic = direction_cosines &\n')
                fid.write(f'    component = {component} &\n')
                fid.write('    create_measure_display = no\n')
                fid.write('!\n')
                fid.write(f'data_element attributes  data_element_name = .{model_name}.{measure_name} &\n')
                fid.write('    color = WHITE\n')
                fid.write('!\n')
 
        # Displacement correction factor variables
        fid.write('! ----- Displacement Correction Factor Variables ----- !\n!\n')
        for _, row in correction_data.iterrows():
            var_name    = str(row['DesignVariableName'])
            transition_x = str(row['transitionXValue'])
            transition_y = str(row['transitionYValue'])
            stiffness    = str(row['Stiffness'])
            fid.write(f'variable create  variable_name = .{model_name}.{var_name} &\n')
            fid.write('    units = "no_units" &\n')
            fid.write('    range = -1.0, 1.0 &\n')
            fid.write('    use_allowed_values = no &\n')
            fid.write('    delta_type = relative &\n')
            fid.write(f'    real_value = ({transition_x}-{transition_y}/Stiffness_{stiffness})\n')
            fid.write('!\n')
 
        # Function measures from Excel
        fid.write('! ----- Function Measures from Excel File ----- !\n!\n')
        for _, row in measure_data.iterrows():
            measure_name     = str(row['Name'])
            measure_function = str(row['Function'])
            measure_unit     = str(row['Unit'])
            fid.write(f'measure create function  measure_name = .{model_name}.{measure_name} &\n')
            fid.write(f'    function = "{measure_function}" &\n')
            fid.write(f'    units = "{measure_unit}" &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
 
        # Constraint measures
        fid.write('! ----- Constriant Measures ----- !\n!\n')
        for _, row in constraint_data.iterrows():
            constrain_name = str(row['Name'])
            target_string  = str(row['TargetDesignVariable'])
            total_flag     = str(row['TotalVsSingleForce']).strip() == 'Total'
 
            fid.write(f'measure create function  measure_name = .{model_name}.Constraint1_{constrain_name}Force &\n')
            if total_flag:
                fid.write(f'    function = ".{model_name}.TotalForce_{constrain_name}-{target_string}+0.05"  &\n')
            elif constrain_name == 'PCL':
                fid.write(f'    function = ".{model_name}.TotalForce_PCL_PM-{target_string}+0.05"  &\n')
            elif constrain_name == 'OPL':
                fid.write(f'    function = ".{model_name}.Force_OPL_PL-{target_string}+0.05"  &\n')
            else:
                fid.write(f'    function = ".{model_name}.Force_{constrain_name}-{target_string}+0.05" &\n')
            fid.write('    units = "force" &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
 
            fid.write(f'measure create function  measure_name = .{model_name}.Constraint2_{constrain_name}Force &\n')
            if total_flag:
                fid.write(f'    function = "{target_string}-0.05-.{model_name}.TotalForce_{constrain_name}" &\n')
            elif constrain_name == 'PCL':
                fid.write(f'    function = "{target_string}-0.05-.{model_name}.TotalForce_PCL_PM"  &\n')
            elif constrain_name == 'OPL':
                fid.write(f'    function = "{target_string}-0.05-.{model_name}.Force_OPL_PL"  &\n')
            else:
                fid.write(f'    function = "{target_string}-0.05-.{model_name}.Force_{constrain_name}"  &\n')
            fid.write('    units = "force" &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
 
        # Constraint optimizers
        fid.write('! ----- Constriant Optimizers ----- !\n!\n')
        for _, row in constraint_data.iterrows():
            optimizer_name = str(row['Name'])
            fid.write(f'optimize constraint create  constraint_name = .{model_name}.OPT_{optimizer_name}_1 &\n')
            fid.write(f'    measure_name = .{model_name}.Constraint1_{optimizer_name}Force &\n')
            fid.write('    output_characteristic = last_value\n')
            fid.write('!\n')
            fid.write(f'optimize constraint create  constraint_name = .{model_name}.OPT_{optimizer_name}_2 &\n')
            fid.write(f'    measure_name = .{model_name}.Constraint2_{optimizer_name}Force &\n')
            fid.write('    output_characteristic = last_value\n')
            fid.write('!\n')
 
        # ── Chapter 8: Force Function Definitions ─────────────────────────────
        fid.write('! ----- New Force Functions ----- !\n!\n')
        for _, row in ligaments_data.iterrows():
            ligament = str(row['LigamentFiber'])
            if any(x in ligament for x in ['OPL', 'PLC', 'PMC', 'FFL']):
                ligament2 = 'sMCL'
            elif 'POL' in ligament:
                ligament2 = ligament.split('_')[0]
            elif 'ACL' in ligament:
                if any(x in ligament for x in ['1', '3']):
                    ligament2 = 'ACL_AM'
                elif '2' in ligament:
                    ligament2 = 'ACL_AL'
                elif any(x in ligament for x in ['4', '5', '6']):
                    ligament2 = 'ACL_PL'
            elif 'PCL' in ligament:
                if any(x in ligament for x in ['5', '6', '7']):
                    ligament2 = 'PCL_AL'
                elif any(x in ligament for x in ['1', '2', '3', '4']):
                    ligament2 = 'PCL_PM'
            else:
                ligament2 = ligament
 
            fid.write(f'measure create function  measure_name =FUN_{ligament} &\n')
            fid.write(f'    function = "STEP(Disp_{ligament}, 0.0, 0.0, 1.0E-003, (DV_A_{ligament2}*((abs(Disp_{ligament}))**DV_B_{ligament2})))" &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
 
        fid.write('! ----- Ligament Force Equations ----- !\n!\n')
        for _, row in ligaments_data.iterrows():
            ligament   = str(row['LigamentFiber'])
            fibers     = row['Fibers']
            stiffness  = str(row['LinearStiffness'])
            toe_region = str(row['ToeRegionCharacteristic'])
 
            multiplier_txt  = f'(1/{fibers})'
            damping_txt     = f'DampingC_Ligaments*StepVR_{ligament}'
            func_txt        = f'(FUN_{ligament})'
 
            toe_step1_txt   = f'Step(Length_{ligament},L0_{ligament},0,L0_{ligament}+0.1,1)'
            toe_step2_txt   = f'Step(Length_{ligament},L0_{ligament}+transitionX_{toe_region},1,L0_{ligament}+transitionX_{toe_region}+0.001,0)'
            toe_region_txt  = f'(-{func_txt}-{damping_txt}) * {toe_step1_txt} * {toe_step2_txt}'
 
            linear_step_txt   = f'Step(Length_{ligament},L0_{ligament}+transitionX_{toe_region} ,0,L0_{ligament}+transitionX_{toe_region}+0.001,1)'
            linear_region_txt = f'(-Stiffness_{stiffness}*(Disp_{ligament}-{toe_region}_DisplacementCorrectionFactor)-{damping_txt})*{linear_step_txt}'
 
            final_text = f'{multiplier_txt}*({toe_region_txt}+{linear_region_txt})'
 
            fid.write(f'force modify direct single_component_force  single_component_force_name = {ligament} &\n')
            fid.write(f'    function = "{final_text}"\n')
            fid.write('!\n')
 
        # ── Chapter 9: Sensors ────────────────────────────────────────────────
        fid.write('! ----- Sensors ----- !\n!\n')
        for _, row in sensor_data.iterrows():
            sensor_name     = str(row['Name'])
            dv_value        = str(row['TerminalDV'])
            control_measure = str(row['ControlMeasure'])
            fid.write(f'executive_control create sensor  sensor_name = .{model_name}.SENSOR_{sensor_name}   &\n')
            fid.write('    compare = ge &\n')
            fid.write(f'    value = ({dv_value}) &\n')
            fid.write('    error = 0.001 &\n')
            fid.write('    codgen = off &\n')
            fid.write('    halt = on &\n')
            fid.write('    print = off &\n')
            fid.write('    restart = off &\n')
            fid.write('    return = off &\n')
            fid.write('    yydump = off &\n')
            fid.write(f'    function = ".{model_name}.{control_measure}"\n')
            fid.write('!\n')
            fid.write(f'executive_control attributes sensor  sensor_name = .{model_name}.SENSOR_{sensor_name} &\n')
            fid.write('    active = off\n')
            fid.write('!\n')
 
        # ── Chapter 10: Density Assignment ───────────────────────────────────
        
        parts = ['Fem', 'FemComp', 'Insert', 'Tray', 'Tib', 'Fib']
        density_dvs = ['BoneDensity', 'FemCompDensity', 'InsertDensity', 'TrayDensity', 'BoneDensity', 'BoneDensity']
 
        fid.write('! ----- Density Assignment(& cm markers off) ----- !\n!\n')
        for part, ddv in zip(parts, density_dvs):
            fid.write(f'part modify rigid mass_properties  part_name = .{model_name}.{part}  &\n')
            fid.write(f'    density = ( .{model_name}.{ddv} )\n')
            fid.write('!\n')
            fid.write(f'marker attributes  marker_name = .{model_name}.{part}.cm  &\n')
            fid.write('    visibility = off\n')
            fid.write('!\n')
 
        if verbose:
            print('--------------------------- Here -------------------------------')
        # ── Write Binary File ───────────────────────────────────────────────
        output_bin_file = f'{self.subject}_C2'
        fid.write('! ----- Write Binary File ----- !\n!\n')
        fid.write(f'file bin write file="{self.bin_dir}/{output_bin_file}.bin" \n!\n')
        
        
        