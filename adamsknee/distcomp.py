import numpy as np
import pandas as pd

class DistCompMixin:
    def distcomp(self, verbose=False, run_adams=True):
        cmd_file = self.cmd_path("step4_DistractionCompression")
        if verbose:
            print(f"Creating simulation CMD file at: {cmd_file}")
        with open(cmd_file, "w", encoding="utf-8") as fid:
            self._write_distcomp(fid, verbose=verbose)
        if run_adams:
            self.run_adams(cmd_file)

    def _write_distcomp(self, fid, iscomp = True, verbose=False):
        
        # ── Read Binary File  ───────────────────────────────────────
        input_bin_file = f'{self.subject}_C3'
        fid.write('! ----- Binary File ----- !\n!\n')
        fid.write(f'file bin read file="{self.bin_dir}/{input_bin_file}.bin" \n!\n')
        
        fid.write('!\n')
        fid.write('! --------------- Distraction --------------------- !\n!\n')
        fid.write('!\n')
        
        # Forces
        fid.write('! ----- Forces ----- !\n')
        fid.write(f'entity attr entity_name=.{self.subject}.MomVV_0d active=off dependents_active=off\n')
        fid.write(f'entity attr entity_name=.{self.subject}.Mom_IE active=off dependents_active=off\n')
        fid.write(f'entity attr entity_name=.{self.subject}.ForceAP_0d active=off dependents_active=off\n')
        fid.write(f'entity attr entity_name=.{self.subject}.ForceCD active=off dependents_active=off\n')
        fid.write('!\n')
        fid.write('!\n')
        
        fid.write('! ------- Axial Motion: Distraction ----- !\n')
        fid.write(f'constraint modify motion motion_name = .{self.subject}.AxialTibTranslation &\n')
        fid.write(f'    joint_name=.{self.subject}.AxialConstraint &\n')
        fid.write(f'    type_of_freedom=translational &\n')
        fid.write(f'    function = "-2*step(time,0,0,.5,1)" &\n')
        fid.write(f'    time_derivative="displacement" &\n')
        fid.write(f'    comments = "" \n')
        fid.write('!\n')
        fid.write('!\n')
        
        # Connections and motions
        fid.write('! ----- Connections and Motions ----- !\n')
        fid.write(f'entity attr entity_name=.{self.subject}.FixFlex_0d active=off dependents_active=off\n')
        fid.write(f'entity attr entity_name=.{self.subject}.FemFlexion active=off dependents_active=off\n')
        fid.write(f'entity attr entity_name=.{self.subject}.AxialConstraint active=on dependents_active=on\n')
        fid.write(f'entity attr entity_name=.{self.subject}.FixFemToGround active=on dependents_active=on\n')
        fid.write(f'entity attr entity_name=.{self.subject}.FemFlexionRotation active=off dependents_active=off\n')
        fid.write(f'entity attr entity_name=.{self.subject}.AxialTibTranslation active=on dependents_active=on\n')
        fid.write(f'entity attr entity_name=.{self.subject}.FixTibToGround active=off dependents_active=off\n')
        fid.write('!\n')
        fid.write('!\n')
        
        
        
        fid.write('! ---------- Run Simulation ----- ! \n')
        fid.write('simulation single_run transient type=dynamic initial_static=no duration=1 step_size=0.01\n!\n')
        
        
        fid.write(f'model copy new_model_name = .{self.subject}_Distraction &\n')
        fid.write(f'analysis = (.{self.subject}.Last_Run) &\n')
        fid.write('frame_number = (101) &\n')
        fid.write('view_name = all &\n')
        fid.write('include_contact_steps="no"\n!\n')
        fid.write(f'model delete model_name=.{self.subject}\n')
        fid.write(f'model display model_name=.{self.subject}_Distraction view_name=.gui.main.*\n')
        fid.write(f'entity modify entity = .{self.subject}_Distraction new = .{self.subject}\n!\n')
        
        if iscomp:
            # Connections and motions
            fid.write('! ----- Connections and Motions ----- !\n!\n')
            fid.write(f'entity attr entity_name=.{self.subject}.FixFlex_0d active=on dependents_active=on\n')
            fid.write(f'entity attr entity_name=.{self.subject}.FemFlexion active=off dependents_active=off\n')
            fid.write(f'entity attr entity_name=.{self.subject}.AxialConstraint active=off dependents_active=off\n')
            fid.write(f'entity attr entity_name=.{self.subject}.FixFemToGround active=on dependents_active=on\n')
            fid.write(f'entity attr entity_name=.{self.subject}.FemFlexionRotation active=off dependents_active=off\n')
            fid.write(f'entity attr entity_name=.{self.subject}.FixTibToGround active=off dependents_active=off\n')
            fid.write(f'entity attr entity_name=.{self.subject}.AxialTibTranslation active=off dependents_active=off\n')
            
            # Forces
            fid.write('! ----- Forces ----- !\n!\n')
            fid.write(f'entity attr entity_name=.{self.subject}.MomVV_0d active=off dependents_active=off\n')
            fid.write(f'entity attr entity_name=.{self.subject}.Mom_IE active=off dependents_active=off\n')
            fid.write(f'entity attr entity_name=.{self.subject}.ForceAP_0d active=off dependents_active=off\n')
            fid.write(f'entity attr entity_name=.{self.subject}.ForceCD active=on dependents_active=on\n')
            
            fid.write('force modify direct single_component_force  &\n')
            fid.write(f'    single_component_force = .{self.subject}.ForceCD  &\n')
            fid.write(f'    function = "400*step(time,0,0,.5,1)"\n')
            fid.write('!\n')
        
            fid.write('! ---------- Run Simulation ----- ! \n')
            fid.write('simulation single_run transient type=dynamic initial_static=no duration=2 step_size=0.01\n')
            
            
            # saved ligaments
            exvar = ['Length_ACL_1','Length_ACL_2','Length_ACL_3','Length_ACL_4','Length_ACL_5','Length_ACL_6',
                     'Length_ALL',
                     'Length_FFL',
                     'Length_LCL',
                     'Length_OPL_DL','Length_OPL_PL',
                     'Length_PCL_1','Length_PCL_2','Length_PCL_3','Length_PCL_4','Length_PCL_5','Length_PCL_6','Length_PCL_7',
                     'Length_PLC_C','Length_PLC_L','Length_PLC_M',
                     'Length_PMC_C','Length_PMC_L','Length_PMC_M',
                     'Length_POL_A','Length_POL_C','Length_POL_P',
                     'Length_sMCL_Sphere2Tib_A','Length_sMCL_Sphere2Tib_C','Length_sMCL_Sphere2Tib_P',
                     'Length_sMCL_WrapDist_A','Length_sMCL_WrapDist_C','Length_sMCL_WrapDist_P',
                     'Length_sMCL_WrapProx_A','Length_sMCL_WrapProx_C','Length_sMCL_WrapProx_P',
                     'Length_sMCL_Wrap_A2C_Sphere','Length_sMCL_Wrap_C2P_Sphere']
            # -----------------------------
            # Ligament Length Plots
            # -----------------------------
            fid.write("xy_plot template modify plot=.plot_1 auto_title=yes auto_subtitle=yes auto_date=yes auto_analysis_name=yes table=no\n")
            fid.write("xy_plot template clear plot=.plot_1\n")
            for idx, var in enumerate(exvar, start=1):
                fid.write(f"xy_plot curve create curve=.plot_1.curve_{idx} create_page=no "
                          f"calculate_axis_limits=no dmeasure=.{self.subject}.{var} "
                          f"imeasure=.{self.subject}.Time run=.{self.subject}.Last_Run auto_axis=UNITS\n")
            fid.write("xy_plot template calculate_axis_limits plot_name=.plot_1\n")
            fid.write('file table write &\n')
            out_lig_file = f'{self.lig_update_dir}/{self.subject}_Ligament_Lengths'
            fid.write(f'   file_name = "{out_lig_file}"  &\n')
            fid.write(f'   plot_name = .plot_1 &\n')
            fid.write(f'   format = spreadsheet \n')
            fid.write('!\n!\n')
            
            # -----------------------------
            # Delete and rename model
            # -----------------------------
            fid.write(f"model copy new_model_name = .{self.subject}_Compression &\n")
            fid.write(f"analysis = (.{self.subject}.Last_Run) &\n")
            fid.write("frame_number = (200) &\n")
            fid.write("view_name = all &\n")
            fid.write('include_contact_steps="no"\n!\n')
            fid.write(f"model delete model_name=.{self.subject}\n")
            fid.write(f"model display model_name=.{self.subject}_Compression view_name=.gui.main.*\n")
            fid.write(f"entity modify entity = .{self.subject}_Compression new = .{self.subject}\n!\n")

            # -----------------------------
            # Save axial constraint info
            # -----------------------------
            fid.write(f'list_info marker marker_name = .{self.subject}.Tib.Axial_Constraint file_name = "{self.model_inputs_dir}/axialConstraint"\n')
            fid.write('!\n!\n')
            
            fid.write('interface dialog undisplay dialog=.gui.info_window\n')
            fid.write('!\n!\n')
            

        # ── Write Binary File ───────────────────────────────────────────────
        output_bin_file = f'{self.subject}_C4'
        fid.write('! ----- Write Binary File ----- !\n!\n')
        fid.write(f'file bin write file="{self.bin_dir}/{output_bin_file}.bin" \n!\n')
