import numpy as np
import pandas as pd
from registrations import read_tfm_file

def _strip_df(df):
    return df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

class SMCLMixin:
    def smcl(self, verbose=False, run_adams=True):
        cmd_file = self.cmd_path("step3_sMCL")
        if verbose:
            print(f"Creating simulation CMD file at: {cmd_file}")
        with open(cmd_file, "w", encoding="utf-8") as fid:
            self._write_smcl(fid, verbose=verbose)
        if run_adams:
            self.run_adams(cmd_file)
    
     
    def _read_smcl_landmarks(self, filepath):
        """
        Reads the sMCL landmark Excel file and returns a dict mapping
        landmark names to numpy xyz arrays. Equivalent to readLandmarksFile.
        """
        df = _strip_df(pd.read_excel(filepath))
        # Expected columns: sMCL_marks, X, Y, Z
        result = {}
        for _, row in df.iterrows():
            result[str(row['sMCL_marks'])] = np.array([row['X'], row['Y'], row['Z']], dtype=float)
        return result
    
    def _write_smcl(self, fid, verbose=False):
        model_name = self.subject
 
        # ── Input files ───────────────────────────────────────────────────────
        femur_xform, _ = read_tfm_file(self.transforms_dir/ "RI.tfm")
        tibia_xform, _ = read_tfm_file(self.transforms_dir/ "RI.tfm")
 
        smcl           = self._read_smcl_landmarks(self.model_inputs_dir / "sMCL_Attachments_FC.xlsx")
        smcl_measure_data = _strip_df(pd.read_excel(self.model_inputs_dir / "sMCLMeasures.xlsx"))
 
        # ── Transform markers into build position ─────────────────────────────
        # Anterior fibers
        ant_femur_pos  = femur_xform @ np.append(smcl['sMCL_A_Fem'],  1)
        ant_sphere_pos = tibia_xform @ np.append(smcl['sMCL_A_Sphere'], 1)
        ant_tibia_pos  = tibia_xform @ np.append(smcl['sMCL_A_Tib'],  1)
 
        # Central fibers
        cen_femur_pos  = femur_xform @ np.append(smcl['sMCL_C_Fem'],  1)
        cen_sphere_pos = tibia_xform @ np.append(smcl['sMCL_C_Sphere'], 1)
        cen_tibia_pos  = tibia_xform @ np.append(smcl['sMCL_C_Tib'],  1)
 
        # Posterior fibers
        pos_femur_pos  = femur_xform @ np.append(smcl['sMCL_P_Fem'],  1)
        pos_sphere_pos = tibia_xform @ np.append(smcl['sMCL_P_Sphere'], 1)
        pos_tibia_pos  = tibia_xform @ np.append(smcl['sMCL_P_Tib'],  1)
 
        # ── L0 lengths at build position ──────────────────────────────────────
        ant_bundle_prox_l0 = np.linalg.norm(ant_sphere_pos[:3] - ant_femur_pos[:3])
        ant_bundle_dist_l0 = np.linalg.norm(ant_sphere_pos[:3] - ant_tibia_pos[:3])
 
        cen_bundle_prox_l0 = np.linalg.norm(cen_sphere_pos[:3] - cen_femur_pos[:3])
        cen_bundle_dist_l0 = np.linalg.norm(cen_sphere_pos[:3] - cen_tibia_pos[:3])
 
        pos_bundle_prox_l0 = np.linalg.norm(pos_sphere_pos[:3] - pos_femur_pos[:3])
        pos_bundle_dist_l0 = np.linalg.norm(pos_sphere_pos[:3] - pos_tibia_pos[:3])
 
        a2c_sphere_l0 = np.linalg.norm(ant_sphere_pos - cen_sphere_pos)
        c2p_sphere_l0 = np.linalg.norm(cen_sphere_pos - pos_sphere_pos)
 
        # ── Fixed parameters ──────────────────────────────────────────────────
        fibers = ['A', 'C', 'P']
 
        marker_names = [
            'Fem.Fem_sMCL_WrapProx_A', 'Fem.Fem_sMCL_WrapProx_C', 'Fem.Fem_sMCL_WrapProx_P',
            'Tib.Tib_sMCL_Sphere_A',   'Tib.Tib_sMCL_Sphere_C',   'Tib.Tib_sMCL_Sphere_P',
            'Tib.Tib_sMCL_WrapDist_A', 'Tib.Tib_sMCL_WrapDist_C', 'Tib.Tib_sMCL_WrapDist_P',
        ]
        marker_locations = np.array([
            ant_femur_pos[:3],  cen_femur_pos[:3],  pos_femur_pos[:3],
            ant_sphere_pos[:3], cen_sphere_pos[:3], pos_sphere_pos[:3],
            ant_tibia_pos[:3],  cen_tibia_pos[:3],  pos_tibia_pos[:3],
        ])
 
        vertical_fiber_l0_values = [
            (ant_bundle_prox_l0, ant_bundle_dist_l0),
            (cen_bundle_prox_l0, cen_bundle_dist_l0),
            (pos_bundle_prox_l0, pos_bundle_dist_l0),
        ]
 
        sphere2sphere_names  = ['sMCL_Wrap_A2C_Sphere', 'sMCL_Wrap_C2P_Sphere']
        sphere2sphere_l0_values = [a2c_sphere_l0, c2p_sphere_l0]
 
        force_names = [
            'WrapProx_A', 'WrapDist_A', 'Sphere2Tib_A',
            'WrapProx_C', 'WrapDist_C', 'Sphere2Tib_C',
            'WrapProx_P', 'WrapDist_P', 'Sphere2Tib_P',
            'Wrap_A2C_Sphere', 'Wrap_C2P_Sphere',
        ]
        origin_markers = [
            'sMCL_WrapSphere_A.cm', 'sMCL_WrapSphere_A.cm', 'sMCL_WrapSphere_A.cm',
            'sMCL_WrapSphere_C.cm', 'sMCL_WrapSphere_C.cm', 'sMCL_WrapSphere_C.cm',
            'sMCL_WrapSphere_P.cm', 'sMCL_WrapSphere_P.cm', 'sMCL_WrapSphere_P.cm',
            'sMCL_WrapSphere_A.cm', 'sMCL_WrapSphere_C.cm',
        ]
        insertion_markers = [
            'Fem.Fem_sMCL_WrapProx_A', 'Tib.Tib_sMCL_WrapDist_A', 'Tib.Tib_sMCL_Sphere_A',
            'Fem.Fem_sMCL_WrapProx_C', 'Tib.Tib_sMCL_WrapDist_C', 'Tib.Tib_sMCL_Sphere_C',
            'Fem.Fem_sMCL_WrapProx_P', 'Tib.Tib_sMCL_WrapDist_P', 'Tib.Tib_sMCL_Sphere_P',
            'sMCL_WrapSphere_C.cm',    'sMCL_WrapSphere_P.cm',
        ]
 
        force_measure_names = [
            'Force_sMCL_Sphere2Tib_A', 'Force_sMCL_WrapDist_A', 'Force_sMCL_WrapProx_A',
            'Force_sMCL_Sphere2Tib_C', 'Force_sMCL_WrapDist_C', 'Force_sMCL_WrapProx_C',
            'Force_sMCL_Sphere2Tib_P', 'Force_sMCL_WrapDist_P', 'Force_sMCL_WrapProx_P',
            'Force_sMCL_A2C_Sphere',   'Force_sMCL_C2P_Sphere',
        ]
        force_measure_objects = [
            'sMCL_Sphere2Tib_A', 'sMCL_WrapDist_A', 'sMCL_WrapProx_A',
            'sMCL_Sphere2Tib_C', 'sMCL_WrapDist_C', 'sMCL_WrapProx_C',
            'sMCL_Sphere2Tib_P', 'sMCL_WrapDist_P', 'sMCL_WrapProx_P',
            'sMCL_Wrap_A2C_Sphere', 'sMCL_Wrap_C2P_Sphere',
        ]
 
        measure_names = [
            'sMCL_WrapProx_A', 'sMCL_WrapProx_C', 'sMCL_WrapProx_P',
            'sMCL_WrapDist_A', 'sMCL_WrapDist_C', 'sMCL_WrapDist_P',
            'sMCL_Sphere2Tib_A', 'sMCL_Sphere2Tib_C', 'sMCL_Sphere2Tib_P',
            'sMCL_Wrap_A2C_Sphere', 'sMCL_Wrap_C2P_Sphere',
        ]
        measure_i_markers = [
            'Fem.Fem_sMCL_WrapProx_A', 'Fem.Fem_sMCL_WrapProx_C', 'Fem.Fem_sMCL_WrapProx_P',
            'Tib.Tib_sMCL_WrapDist_A', 'Tib.Tib_sMCL_WrapDist_C', 'Tib.Tib_sMCL_WrapDist_P',
            'Tib.Tib_sMCL_Sphere_A',   'Tib.Tib_sMCL_Sphere_C',   'Tib.Tib_sMCL_Sphere_P',
            'sMCL_WrapSphere_A.cm',    'sMCL_WrapSphere_C.cm',
        ]
        measure_j_markers = [
            'sMCL_WrapSphere_A.cm', 'sMCL_WrapSphere_C.cm', 'sMCL_WrapSphere_P.cm',
            'sMCL_WrapSphere_A.cm', 'sMCL_WrapSphere_C.cm', 'sMCL_WrapSphere_P.cm',
            'sMCL_WrapSphere_A.cm', 'sMCL_WrapSphere_C.cm', 'sMCL_WrapSphere_P.cm',
            'sMCL_WrapSphere_C.cm', 'sMCL_WrapSphere_P.cm',
        ]
 
        vertical_ligament_names   = ['sMCL_WrapProx_A', 'sMCL_WrapDist_A',
                                      'sMCL_WrapProx_C', 'sMCL_WrapDist_C',
                                      'sMCL_WrapProx_P', 'sMCL_WrapDist_P']
        spherical_ligament_names  = ['sMCL_Sphere2Tib_A', 'sMCL_Sphere2Tib_C', 'sMCL_Sphere2Tib_P',
                                      'sMCL_Wrap_A2C_Sphere', 'sMCL_Wrap_C2P_Sphere']
 
        constraint_names  = ['sMCL_WrapProx', 'sMCL_WrapDist']
        optimizer_names   = ['sMCL_WrapProx', 'sMCL_WrapDist']
 
        joint_marker_names = [
            'Tib.PlanarJoint_Sphere2Tib_A',
            'sMCL_WrapSphere_A.PlanarJoint_Sphere2Tib_A',
            'Tib.PlanarJoint_Sphere2Tib_C',
            'sMCL_WrapSphere_C.PlanarJoint_Sphere2Tib_C',
            'Tib.PlanarJoint_Sphere2Tib_P',
            'sMCL_WrapSphere_P.PlanarJoint_Sphere2Tib_P',
        ]
        joint_marker_locations = [
            ant_sphere_pos[:3], ant_sphere_pos[:3],
            cen_sphere_pos[:3], cen_sphere_pos[:3],
            pos_sphere_pos[:3], pos_sphere_pos[:3],
        ]

        # ── Read Binary File  ───────────────────────────────────────
        input_bin_file = f'{self.subject}_C2'
        fid.write('! ----- Binary File ----- !\n!\n')
        fid.write(f'file bin read file="{self.bin_dir}/{input_bin_file}.bin" \n!\n')
        
        
        # ── Markers ───────────────────────────────────────────────────────────
        fid.write('! ----- sMCL Markers ----- !\n!\n')
        for marker, loc in zip(marker_names, marker_locations):
            fid.write(f'marker create marker_name =.{model_name}.{marker} &\n')
            fid.write(f'    location = {loc[0]:f},{loc[1]:f},{loc[2]:f} &\n')
            fid.write('   orientation = 0.0d, 0.0d, 0.0d\n')
            fid.write(f'marker attributes  marker_name =.{model_name}.{marker} &\n')
            fid.write('    visibility = off \n')
            fid.write('!\n')
 
        # ── Spheres ───────────────────────────────────────────────────────────
        fid.write('! ----- sMCL Spheres ----- !\n!\n')
        for fiber in fibers:
            fid.write(f'part create rigid_body name_and_position part_name = sMCL_WrapSphere_{fiber}  \n')
            fid.write(f'marker create  marker_name = .{model_name}.sMCL_WrapSphere_{fiber}.sMCL_SphereCenter_{fiber}  &\n')
            fid.write(f'    location = (LOC_GLOBAL( {{0.0, 0, 0.0}} , .{model_name}.Tib.Tib_sMCL_Sphere_{fiber} ) ) &\n')
            fid.write('    orientation = 0.0, 0.0, 0.0 \n')
            fid.write('!\n')
            fid.write(f'marker attributes marker_name = .{model_name}.sMCL_WrapSphere_{fiber}.sMCL_SphereCenter_{fiber}  &\n')
            fid.write('    visibility = off\n')
            fid.write('!\n')
            fid.write(f'geometry create shape ellipsoid  ellipsoid_name = .{model_name}.sMCL_WrapSphere_{fiber}.ELLIPSOID_1 &\n')
            fid.write(f'    center_marker = .{model_name}.sMCL_WrapSphere_{fiber}.sMCL_SphereCenter_{fiber} &\n')
            fid.write('    x_scale_factor = 1.0 &\n')
            fid.write('    y_scale_factor = 1.0 &\n')
            fid.write('    z_scale_factor = 1.0\n')
            fid.write('!\n')
            fid.write(f'part attributes part_name = sMCL_WrapSphere_{fiber} &\n')
            fid.write('    color = CYAN &\n')
            fid.write('    name_visibility = off\n')
            fid.write('!\n')
            fid.write(f'part modify rigid mass_properties  part_name = .{model_name}.sMCL_WrapSphere_{fiber} &\n')
            fid.write(f'    density = (.{model_name}.sMCLSphereDensity)\n')
            fid.write('!\n')
            fid.write(f'marker attributes  marker_name = .{model_name}.sMCL_WrapSphere_{fiber}.cm  &\n')
            fid.write('    visibility = off\n')
            fid.write('!\n')
 
        # ── Force elements ────────────────────────────────────────────────────
        fid.write('! ----- sMCL Forces ----- !\n!\n')
        for force, orig, insert in zip(force_names, origin_markers, insertion_markers):
            fid.write('force create direct single_component_force &\n')
            fid.write(f'single_component_force_name = sMCL_{force} &\n')
            fid.write('type_of_freedom = translational &\n')
            fid.write(f'i_marker_name = {orig} &\n')
            fid.write(f'j_marker_name = {insert} &\n')
            fid.write('action_only = off &\n')
            fid.write('function = ""\n')
            fid.write('!\n')
            fid.write('force attributes &\n')
            fid.write(f'   force_name = sMCL_{force} &\n')
            fid.write('   size_of_icons = 10.0&\n')
            fid.write('   name_visibility = off\n')
            fid.write('!\n')
 
        # ── L0 Variables ──────────────────────────────────────────────────────
        fid.write('! ----- sMCL L0 ----- !\n!\n')
        for fiber, (prox_l0, dist_l0) in zip(fibers, vertical_fiber_l0_values):
            fid.write(f'variable create  variable_name = .{model_name}.L0_sMCL_WrapProx_{fiber} &\n')
            fid.write('    units = "length" &\n')
            fid.write('    range = -1.0, 1.0 &\n')
            fid.write('    use_allowed_values = no &\n')
            fid.write('    delta_type = relative &\n')
            fid.write(f'    real_value = (Percent_L0_sMCL_Prox * {prox_l0:f} )')
            fid.write('!\n')
 
            fid.write(f'variable create  variable_name = .{model_name}.L0_sMCL_WrapDist_{fiber} &\n')
            fid.write('    units = "length" &\n')
            fid.write('    range = -1.0, 1.0 &\n')
            fid.write('    use_allowed_values = no &\n')
            fid.write('    delta_type = relative &\n')
            fid.write(f'    real_value = (Percent_L0_sMCL_Dist * {dist_l0:f} )')
            fid.write('!\n')
 
            fid.write(f'variable create  variable_name = .{model_name}.L0_sMCL_Sphere2Tib_{fiber} &\n')
            fid.write('    units = "length" &\n')
            fid.write('    range = -1.0, 1.0 &\n')
            fid.write('    use_allowed_values = no &\n')
            fid.write('    delta_type = relative &\n')
            fid.write('    real_value = 1\n')
            fid.write('!\n')
 
        for name, l0_value in zip(sphere2sphere_names, sphere2sphere_l0_values):
            fid.write(f'variable create  variable_name = L0_{name} &\n')
            fid.write('    units = "no_units" &\n')
            fid.write('    range = -1.0, 1.0 &\n')
            fid.write('    use_allowed_values = no &\n')
            fid.write('    delta_type = relative &\n')
            fid.write(f'    real_value = {l0_value:f}\n')
            fid.write('!\n')
 
        # ── Force measures ────────────────────────────────────────────────────
        fid.write('! ----- sMCL Force Measures ----- !\n!\n')
        for measure, obj in zip(force_measure_names, force_measure_objects):
            fid.write(f'measure create object  measure_name = {measure} &\n')
            fid.write('    from_first = yes &\n')
            fid.write(f'    object = {obj}  &\n')
            fid.write('    characteristic = element_force &\n')
            fid.write('    component = mag_component &\n')
            fid.write('    create_measure_display = no\n')
            fid.write(f'data_element attributes  data_element_name = {measure} &\n')
            fid.write('    color = WHITE\n')
            fid.write('!\n')
 
        # ── Length, VR, Displacement, Strain, StepVR measures ─────────────────
        fid.write('! ----- Length Measures ----- !\n!\n')
        for name, i_m, j_m in zip(measure_names, measure_i_markers, measure_j_markers):
            fid.write('measure create function &\n')
            fid.write(f'    measure_name = Length_{name} &\n')
            fid.write(f'    function = "DM({i_m},{j_m})" &\n')
            fid.write('    units = length &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
 
        fid.write('! ----- VR Measures ----- !\n!\n')
        for name, i_m, j_m in zip(measure_names, measure_i_markers, measure_j_markers):
            fid.write(f'measure create function  measure_name =VR_{name} &\n')
            fid.write(f'    function = "VR({i_m},{j_m})" &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
 
        fid.write('! ----- Dispalcement Measures ----- !\n!\n')
        for name in measure_names:
            fid.write('measure create function &\n')
            fid.write(f'    measure_name = Disp_{name} &\n')
            fid.write(f'    function = "(Length_{name}-L0_{name})" &\n')
            fid.write('    units = length &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
 
        # Strain measures — skipped for indices 6,7,8 (0-based), i.e. Sphere2Tib fibers
        fid.write('! ----- Strain Measures ----- !\n!\n')
        for i, name in enumerate(measure_names):
            if i in (6, 7, 8):   # MATLAB iMeasure 7,8,9 → 0-based 6,7,8
                continue
            fid.write('measure create function &\n')
            fid.write(f'    measure_name = Strain_{name} &\n')
            fid.write(f'    function = "(Disp_{name}/L0_{name})" &\n')
            fid.write('    units = no_units &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
 
        fid.write('! ----- StepVR Measures ----- !\n!\n')
        for name in measure_names:
            fid.write('!\n')
            fid.write('measure create function  &\n')
            fid.write(f'   measure_name =.{model_name}.StepVR_{name} &\n')
            fid.write(f'   function = "VR_{name}*Step(VR_{name}, 0, 0, VR_{name} +0.1,1)" &\n')
            fid.write('   create_measure_display = no\n')
 
        # ── Function measures from sMCL measures Excel file ───────────────────
        fid.write('! ----- Function Measures from sMCL Measures Excel File ----- !\n!\n')
        for _, row in smcl_measure_data.iterrows():
            measure_name     = str(row['MeasureName'])
            unit             = str(row['Unit'])
            measure_function = str(row['Function'])
            fid.write('!\n')
            fid.write('measure create function  &\n')
            fid.write(f'    measure_name = .{model_name}.{measure_name}   &\n')
            fid.write(f'    function = "{measure_function}" &\n')
            fid.write(f'    units = "{unit}"  &\n')
            fid.write('    create_measure_display = no\n')
 
        # ── New force functions ───────────────────────────────────────────────
        fid.write('! ----- New Force Functions ----- !\n!\n')
        for ligament in vertical_ligament_names:
            temp     = ligament.split('_')
            ligament2 = temp[0]   # e.g. 'sMCL'
            fid.write(f'measure create function  measure_name =FUN_{ligament} &\n')
            fid.write(f'    function = "STEP(Disp_{ligament}, 0.0, 0.0, 1.0E-003, (DV_A_{ligament2}*((abs(Disp_{ligament}))**DV_B_{ligament2})))" &\n')
            fid.write('    create_measure_display = no\n')
            fid.write('!\n')
 
        # ── Force equations: vertical fibers ─────────────────────────────────
        fid.write('! ----- Force Equations for Vertical Fibers ----- !\n!\n')
        for ligament in vertical_ligament_names:
            multiplier_txt    = '(2/3)'
            damping_txt       = f'DampingC_Ligaments*StepVR_{ligament}'
            func_txt          = f'(FUN_{ligament})'
            toe_step1_txt     = f'Step(Length_{ligament},L0_{ligament},0,L0_{ligament}+0.1,1)'
            toe_step2_txt     = f'Step(Length_{ligament},L0_{ligament}+transitionX_sMCL ,1, L0_{ligament}+transitionX_sMCL +0.001,0)'
            toe_region_txt    = f'(-{func_txt}-{damping_txt})*{toe_step1_txt}*{toe_step2_txt}'
            linear_step_txt   = f'Step(Length_{ligament},L0_{ligament}+transitionX_sMCL,0,L0_{ligament}+transitionX_sMCL+0.001,1)'
            linear_region_txt = f'(-Stiffness_sMCL*(Disp_{ligament}-sMCL_DisplacementCorrectionFactor)-{damping_txt})*{linear_step_txt}'
            final_text        = f'{multiplier_txt}*({toe_region_txt}+{linear_region_txt})'
 
            fid.write('force modify direct single_component_force &\n')
            fid.write(f'    single_component_force_name = {ligament} &\n')
            fid.write(f'    function = "{final_text}"\n')
            fid.write('!\n')
 
        # ── Force equations: sphere-to-sphere & sphere-to-tib fibers ──────────
        fid.write('! ----- Force Equations for Sphere 2 Sphere & Sphere 2 Tib Fibers ----- !\n!\n')
        for ligament in spherical_ligament_names:
            multiplier_txt    = '(1)'
            damping_txt       = f'DampingC_Wraps*StepVR_{ligament}'
            linear_step_txt   = f'Step(Length_{ligament},L0_{ligament},0,L0_{ligament}+0.1,1)'
            linear_region_txt = f'(-Stiffness_sMCL_Sphere2Tib*(Disp_{ligament})-{damping_txt})*{linear_step_txt}'
            final_text        = f'{multiplier_txt}*({linear_region_txt}-{damping_txt})*{linear_step_txt}'
 
            fid.write('force modify direct single_component_force &\n')
            fid.write(f'    single_component_force_name = {ligament} &\n')
            fid.write(f'    function = "{final_text}"\n')
            fid.write('!\n')
 
        # ── Constraint measures & optimizers ──────────────────────────────────
        fid.write('! ----- sMCL Constraint Measures ----- !\n!\n')
        for cname in constraint_names:
            fid.write('measure create function  &\n')
            fid.write(f'    measure_name = .{model_name}.Constraint1_{cname}Force   &\n')
            fid.write(f'    function = ".{model_name}.TotalForce_{cname}-Target_WrapsMCL+0.05"  &\n')
            fid.write('    units = "force"  &\n')
            fid.write('    create_measure_display = no \n')
            fid.write('!\n')
            fid.write('measure create function  &\n')
            fid.write(f'    measure_name = .{model_name}.Constraint2_{cname}Force   &\n')
            fid.write(f'    function = "Target_WrapsMCL-0.05-.{model_name}.TotalForce_{cname}"  &\n')
            fid.write('    units = "force"  &\n')
            fid.write('    create_measure_display = no \n')
            fid.write('!\n')
 
        fid.write('! ----- sMCL Optimizers ----- !\n!\n')
        for opt_name in optimizer_names:
            fid.write('optimize constraint create &\n')
            fid.write(f'    constraint_name = .{model_name}.OPT_{opt_name}_1 &\n')
            fid.write(f'    measure_name = .{model_name}.Constraint1_{opt_name}Force &\n')
            fid.write('    output_characteristic = last_value\n')
            fid.write('!\n')
            fid.write('optimize constraint create &\n')
            fid.write(f'    constraint_name = .{model_name}.OPT_{opt_name}_2 &\n')
            fid.write(f'    measure_name = .{model_name}.Constraint2_{opt_name}Force &\n')
            fid.write('    output_characteristic = last_value\n')
            fid.write('!\n')
 
        # ── Sensor ────────────────────────────────────────────────────────────
        fid.write('! ----- sMCL Sensor ----- !\n!\n')
        fid.write('executive_control create sensor &\n')
        fid.write(f'    sensor_name = .{model_name}.SENSOR_WrapsMCL &\n')
        fid.write('    compare = ge &\n')
        fid.write('    value = (Failure_sMCL) &\n')
        fid.write('    error = 0.001 &\n')
        fid.write('    codgen = off &\n')
        fid.write('    halt = on &\n')
        fid.write('    print = off &\n')
        fid.write('    restart = off &\n')
        fid.write('    return = off &\n')
        fid.write('    yydump = off &\n')
        fid.write(f'    function = ".{model_name}.TotalForce_WrapsMCL"\n')
        fid.write('!\n')
        fid.write('executive_control attributes sensor &\n')
        fid.write(f'    sensor_name = .{model_name}.SENSOR_WrapsMCL &\n')
        fid.write('    active = off\n')
        fid.write('!\n')
 
        # ── Planar constraint markers ─────────────────────────────────────────
        fid.write('! ----- Planar Constraint Markers ----- !\n!\n')
        for marker, loc in zip(joint_marker_names, joint_marker_locations):
            fid.write(f'marker create marker_name=.{model_name}.{marker} &\n')
            fid.write(f'    location = {loc[0]:f} , {loc[1]:f} , {loc[2]:f}  &\n')
            fid.write('    orientation = 90.0D, 0.0d, 0.0d\n')
            fid.write('!\n')
            fid.write(f'marker attributes marker_name = .{model_name}.{marker} &\n')
            fid.write('    visibility = off\n')
            fid.write('!\n')
 
        # ── Planar constraints ────────────────────────────────────────────────
        fid.write('! ----- Planar Constraints ----- !\n!\n')
        for fiber in fibers:
            joint_name    = f'PlanarJoint_Sphere2Tib_{fiber}'
            joint_i_marker = f'.{model_name}.sMCL_WrapSphere_{fiber}.PlanarJoint_Sphere2Tib_{fiber}'
            joint_j_marker = f'.{model_name}.Tib.PlanarJoint_Sphere2Tib_{fiber}'
            fid.write('constraint create joint planar &\n')
            fid.write(f'    joint_name = {joint_name} &\n')
            fid.write(f'    i_marker_name = {joint_i_marker} &\n')
            fid.write(f'    j_marker_name =  {joint_j_marker}\n')
            fid.write('constraint attributes &\n')
            fid.write(f'    constraint_name = {joint_name} &\n')
            fid.write('    visibility = off\n')
            fid.write('!\n')
 
        # ── Simulation script, objective & gravity ────────────────────────────
        fid.write('! ----- Final Touch: Simulation Script, Objective Optimizer, & Deactivate Gravity ----- !\n!\n')
        fid.write('simulation script create  &\n')
        fid.write(f'    sim_script_name = .{model_name}.LigL0_OptimizationScript  &\n')
        fid.write('    commands =   &\n')
        fid.write(f'    "simulation single_run transient type=dynamic initial_static=no end_time=0.75 step_size=5.0E-003 model_name=.{model_name}"\n')
        fid.write('!\n')
        fid.write('optimize objective create  &\n')
        fid.write(f'     objective_name = .{model_name}.OBJECTIVE_SummedForceErrors  &\n')
        fid.write(f'     measure_name = .{model_name}.OBJ_SummedForceErrors  & \n')
        fid.write('     output_characteristic = last_value \n')
        fid.write('!\n')
        fid.write(f'!entity attr entity_name=.{model_name}.gravity active=off dependents_active=off\n')
        fid.write('!\n')
        
        
        # Deactivate the cruciate ligaments
        fid.write('! ----- Deactivate ACL fibers forces and their dependents ----- !\n!\n')
        cruciate_liagments = ['ACL_1','ACL_2','ACL_3','ACL_4','ACL_5','ACL_6']
        for ligament in cruciate_liagments:
            fid.write(f'entity attr entity_name=.{model_name}.{ligament} active=off dependents_active=off\n')
        fid.write(f'entity attr entity_name=.{model_name}.Constraint1_ACLForce active=off dependents_active=off\n')
        fid.write(f'entity attr entity_name=.{model_name}.Constraint2_ACLForce active=off dependents_active=off\n')
        
        
        # ── Write Binary File ───────────────────────────────────────────────
        output_bin_file = f'{self.subject}_C3'
        fid.write('! ----- Write Binary File ----- !\n!\n')
        fid.write(f'file bin write file="{self.bin_dir}/{output_bin_file}.bin" \n!\n')