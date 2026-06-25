import os
import pandas as pd
import numpy as np

from registrations import read_tfm_file, tfm2euler123

class UpdateLigsMixin:
    def updateligs(self, verbose=False, run_adams=True):
        cmd_file = self.cmd_path("step5_UpdateLigaments")
        if verbose:
            print(f"Creating simulation CMD file at: {cmd_file}")
        with open(cmd_file, "w", encoding="utf-8") as fid:
            self._write_updateligs(fid, verbose=verbose)
        if run_adams:
            self.run_adams(cmd_file)
        
    def _write_updateligs(self, fid, verbose=False):
        
        # -----------------------------
        # Clean up the Lig Update Directory
        # -----------------------------
        print("  Saving and Reading the Ligament Length ")
        os.unlink(f"{self.lig_update_dir}/{self.subject}_Ligament_Lengths.xlsx") if \
            os.path.exists(f"{self.lig_update_dir}/{self.subject}_Ligament_Lengths.xlsx") else None
        os.unlink(f"{self.lig_update_dir}/Ligament_Lengths.xlsx") if \
            os.path.exists(f"{self.lig_update_dir}/Ligament_Lengths.xlsx") else None
        
        lig_df = pd.read_csv(
            self.lig_update_dir / f"{self.subject}_Ligament_Lengths.tab", sep='\t', skiprows=1)
        lig_df.to_excel(f"{self.lig_update_dir}/{self.subject}_Ligament_Lengths.xlsx", index=False)
        
        
        # ── Read Binary File  ───────────────────────────────────────
        input_bin_file = f'{self.subject}_C4'
        fid.write('! ----- Binary File ----- !\n!\n')
        fid.write(f'file bin read file="{self.bin_dir}/{input_bin_file}.bin" \n!\n')
        
        
        # OPL_DL, OPL_PL
        ligs = ['OPL_DL', 'OPL_PL']
        for lig in ligs:
            fid.write(f'variable modify  & \n')
            fid.write(f'\t variable_name = .{self.subject}.L0_{lig}  & \n')
            length_lig = 'Length_'+lig
            fid.write(f'\t real_value = (.{self.subject}.Percent_L0_PostLatCorner * Percent_L0_OPL * {lig_df.loc[lig_df.index[-1], length_lig]:.6f} )\n')
            fid.write('!\n')
        
        ligs = ['FFL']
        for lig in ligs:
            fid.write(f'variable modify  & \n')
            fid.write(f'\t variable_name = .{self.subject}.L0_{lig}  & \n')
            length_lig = 'Length_'+lig
            fid.write(f'\t real_value = (.{self.subject}.Percent_L0_PostLatCorner * Percent_L0_FFL * {lig_df.loc[lig_df.index[-1], length_lig]:.6f} )\n')
            fid.write('!\n')
            
        ligs = ['PLC_L', 'PLC_C', 'PLC_M']
        for lig in ligs:
            fid.write(f'variable modify  & \n')
            fid.write(f'\t variable_name = .{self.subject}.L0_{lig}  & \n')
            length_lig = 'Length_'+lig
            fid.write(f'\t real_value = (.{self.subject}.Percent_L0_PostCapsule * Percent_L0_PLC * {lig_df.loc[lig_df.index[-1], length_lig]:.6f} )\n')
            fid.write('!\n')
            
        ligs = ['PMC_L', 'PMC_C', 'PMC_M']
        for lig in ligs:
            fid.write(f'variable modify  & \n')
            fid.write(f'\t variable_name = .{self.subject}.L0_{lig}  & \n')
            length_lig = 'Length_'+lig
            fid.write(f'\t real_value = (.{self.subject}.Percent_L0_PostCapsule * Percent_L0_PMC * {lig_df.loc[lig_df.index[-1], length_lig]:.6f} )\n')
            fid.write('!\n')
        
        # Cruciate Ligaments
        ligs = [('PCL_1', 'PCL_PM'), ('PCL_2', 'PCL_PM'), ('PCL_3', 'PCL_PM'), ('PCL_4', 'PCL_PM'), 
                ('PCL_5', 'PCL_AL'), ('PCL_6', 'PCL_AL'), ('PCL_7', 'PCL_AL')]
        for lig, bundle in ligs:
            fid.write(f'variable modify  & \n')
            fid.write(f'\t variable_name = .{self.subject}.L0_{lig}  & \n')
            length_lig = 'Length_'+lig
            fid.write(f'\t real_value = (.{self.subject}.Percent_L0_PCL * Percent_L0_{bundle} * {lig_df.loc[lig_df.index[-1], length_lig]:.6f} )\n')
            fid.write('!\n')
            
        ligs = ['ACL_1', 'ACL_2', 'ACL_3', 'ACL_4', 'ACL_5', 'ACL_6']
        for lig in ligs:
            fid.write(f'variable modify  & \n')
            fid.write(f'\t variable_name = .{self.subject}.L0_{lig}  & \n')
            length_lig = 'Length_'+lig
            fid.write(f'\t real_value = (.{self.subject}.Percent_L0_ACL * {lig_df.loc[lig_df.index[-1], length_lig]:.6f} )\n')
            fid.write('!\n')
        
        
        # Collateral Ligament
        ligs = ['sMCL_WrapProx_A', 'sMCL_WrapProx_C', 'sMCL_WrapProx_P']
        for lig in ligs:
            fid.write(f'variable modify  & \n')
            fid.write(f'\t variable_name = .{self.subject}.L0_{lig}  & \n')
            length_lig = 'Length_'+lig
            fid.write(f'\t real_value = (.{self.subject}.Percent_L0_sMCL_Prox * {lig_df.loc[lig_df.index[-1], length_lig]:.6f} )\n')
            fid.write('!\n')
            
        ligs = ['sMCL_WrapDist_A', 'sMCL_WrapDist_C', 'sMCL_WrapDist_P']
        for lig in ligs:
            fid.write(f'variable modify  & \n')
            fid.write(f'\t variable_name = .{self.subject}.L0_{lig}  & \n')
            length_lig = 'Length_'+lig
            fid.write(f'\t real_value = (.{self.subject}.Percent_L0_sMCL_Dist * {lig_df.loc[lig_df.index[-1], length_lig]:.6f} )\n')
            fid.write('!\n')
        
        ligs = ['sMCL_Sphere2Tib_A', 'sMCL_Sphere2Tib_C', 'sMCL_Sphere2Tib_P']
        for lig in ligs:
            fid.write(f'variable modify  & \n')
            fid.write(f'\t variable_name = .{self.subject}.L0_{lig}  & \n')
            length_lig = 'Length_'+lig
            fid.write(f'\t real_value = ( {lig_df.loc[lig_df.index[-1], length_lig]:.6f} )\n')
            fid.write('!\n')
        
        ligs = ['sMCL_Wrap_A2C_Sphere', 'sMCL_Wrap_C2P_Sphere']
        for lig in ligs:
            fid.write(f'variable modify  & \n')
            fid.write(f'\t variable_name = .{self.subject}.L0_{lig}  & \n')
            length_lig = 'Length_'+lig
            fid.write(f'\t real_value = ( {lig_df.loc[lig_df.index[-1], length_lig]:.6f} )\n')
            fid.write('!\n')
        
        ligs = ['POL_A', 'POL_C', 'POL_P']
        for lig in ligs:
            fid.write(f'variable modify  & \n')
            fid.write(f'\t variable_name = .{self.subject}.L0_{lig}  & \n')
            length_lig = 'Length_'+lig
            fid.write(f'\t real_value = (.{self.subject}.Percent_L0_POL * {lig_df.loc[lig_df.index[-1], length_lig]:.6f} )\n')
            fid.write('!\n')
        
        ligs = ['LCL']
        for lig in ligs:
            fid.write(f'variable modify  & \n')
            fid.write(f'\t variable_name = .{self.subject}.L0_{lig}  & \n')
            length_lig = 'Length_'+lig
            fid.write(f'\t real_value = (.{self.subject}.Percent_L0_LCL * {lig_df.loc[lig_df.index[-1], length_lig]:.6f} )\n')
            fid.write('!\n')
        
        fid.write('!\n')
        fid.write('! ----------- Prepare for Ligament Optimization ----- !\n')
        fid.write('!\n')
        dvars = ['ACL', 'FFL', 'LCL', 'OPL', 'PCL_PM', 'PCL_AL', 'PLC', 'PMC', 'POL', 'sMCL_Dist', 'sMCL_Prox']
        dvals = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        for dvar, dval in zip(dvars, dvals):
            fid.write(f'variable modify  &\n')
            fid.write(f' variable_name = .{self.subject}.Percent_L0_{dvar}  &\n')
            fid.write(f' real_value = {dval} \n!\n')

        target_ligs = ['Target_WrapsMCL']
        target_ligtens = 4
        for klig in target_ligs:
            fid.write(f'variable modify  &\n')
            fid.write(f' variable_name = .{self.subject}.{klig} &\n')
            fid.write(f' real_value = {target_ligtens} \n!\n')
        
        
        fid.write('force modify direct single_component_force  &\n')
        fid.write(f'    single_component_force = .{self.subject}.ForceCD  &\n')
        fid.write(f'    function = "10*step(time,0,0,.5,1)"\n')
        fid.write('!\n')
        
        
        TwrtF, euler_xyz_TwrtF  =    read_tfm_file( self.transforms_dir / "TwrtF.tfm")
        # PD marker
        pd_marker_xform = TwrtF @ np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
        if verbose:
            print(f"PD Marker transformation matrix:\n{pd_marker_xform}")
        pd_marker_xyz123 = tfm2euler123(pd_marker_xform)
        
        if verbose:
            print(f"PD Marker location (x,y,z): {pd_marker_xyz123[:3]}")
            print(f"PD Marker orientation (x,y,z): {pd_marker_xyz123[3:]}")
        
        # readcell behavior → read each line and split into tokens
        table = []
        with open(self.model_inputs_dir / "axialConstraint", "r") as f:
            for line in f:
                # split on whitespace
                table.append(line.strip().split())
        loc_raw = table[9][2]
        loc_val = loc_raw.rstrip(",")
        # ----------------------------------------
        # Write location line
        # ----------------------------------------
        fid.write(
            f"marker modify marker_name=.{self.subject}.Tib.Axial_Constraint "
            f"location = {loc_val},0,0\n"
        )
        fid.write(
            f"marker modify marker_name=.{self.subject}.Tib.Axial_Constraint "
            f"orientation = {pd_marker_xyz123[3]:f},{pd_marker_xyz123[4]:f},{pd_marker_xyz123[5]:f}\n"
        )
                                                                
                                                                                                
        # ── Write Binary File ───────────────────────────────────────────────
        output_bin_file = f'{self.subject}_C5'
        fid.write('! ----- Write Binary File ----- !\n!\n')
        fid.write(f'file bin write file="{self.bin_dir}/{output_bin_file}.bin" \n!\n')