import numpy as np
import pandas as pd

class PreOptimizationMixin:
    def preoptimization(self, verbose=False, run_adams=True):
        cmd_file = self.cmd_path("step5A_preoptimization")
        if verbose:
            print(f"Creating simulation CMD file at: {cmd_file}")
        with open(cmd_file, "w", encoding="utf-8") as fid:
            self._write_preoptimization(fid, verbose=verbose)
        if run_adams:
            self.run_adams(cmd_file)
            self._write_forces(verbose=verbose)
            
    
    def _write_preoptimization(self, fid, verbose=False):
        vars_df = pd.read_excel(self.model_inputs_dir / "Optimization Initial Guesses.xlsx")
        
        # ── Read Binary File  ───────────────────────────────────────
        input_bin_file = f'{self.subject}_C5'
        fid.write('! ----- Binary File ----- !\n!\n')
        fid.write(f'file bin read file="{self.bin_dir}/{input_bin_file}.bin" \n!\n')
        fid.write('interface dialog undisplay dialog=.gui.info_window\n')
        fid.write('!\n!\n')
        for _, row in vars_df.iterrows():
            var_name = row['Name']
            var_value = row['Value']
            fid.write(f'variable modify  &\n')
            fid.write(f' variable_name = .{self.subject}.{var_name}  &\n')
            fid.write(f' real_value = {var_value} \n!\n')
        
        fid.write('force modify direct single_component_force  &\n')
        fid.write(f'    single_component_force = .{self.subject}.ForceCD  &\n')
        fid.write(f'    function = "10*step(time,0,0,.5,1)"\n')
        fid.write('!\n')
        
        fid.write('! ---------- Run Simulation ----- ! \n')
        fid.write('simulation single_run transient type=dynamic initial_static=no duration=1 step_size=0.01\n')
        
        # saved ligaments
        forcevars = ['Force_FFL', 'Force_LCL', 'Force_OPL_PL', 'OBJ_SummedForceErrors', 'TotalForce_PCL_PM' , 'TotalForce_PCL_AL' ,
                     'TotalForce_PLC', 'TotalForce_PMC', 'TotalForce_POL', 'TotalForce_sMCL_WrapDist', 'TotalForce_sMCL_WrapProx' ,
                     'Force_ALL' ]
        # -----------------------------
        # Ligament Length Plots
        # -----------------------------
        fid.write("xy_plot template modify plot=.plot_1 auto_title=yes auto_subtitle=yes auto_date=yes auto_analysis_name=yes table=no\n")
        fid.write("xy_plot template clear plot=.plot_1\n")
        for idx, var in enumerate(forcevars, start=1):
            fid.write(f"xy_plot curve create curve=.plot_1.curve_{idx} create_page=no "
                        f"calculate_axis_limits=no dmeasure=.{self.subject}.{var} "
                        f"imeasure=.{self.subject}.Time run=.{self.subject}.Last_Run auto_axis=UNITS\n")
        fid.write("xy_plot template calculate_axis_limits plot_name=.plot_1\n")
        fid.write('file table write &\n')
        out_lig_file = f'{self.lig_update_dir}/{self.subject}_Ligament_Forces'
        fid.write(f'   file_name = "{out_lig_file}"  &\n')
        fid.write(f'   plot_name = .plot_1 &\n')
        fid.write(f'   format = spreadsheet \n')
        fid.write('!\n!\n')
        
        
            
        # ── Write Binary File ───────────────────────────────────────────────
        output_bin_file = f'{self.subject}_C5_preOpt'
        fid.write('! ----- Write Binary File ----- !\n!\n')
        fid.write(f'file bin write file="{self.bin_dir}/{output_bin_file}.bin" \n!\n')
        
    def _write_forces(self, verbose=False):
        lig_df = pd.read_csv(
            self.lig_update_dir / f"{self.subject}_Ligament_Forces.tab", sep='\t', skiprows=1)
        
        lig_def_names = lig_df.columns[1:]  # Exclude 'Time' column
        
        for col in lig_def_names:
            lig_df[col] = pd.to_numeric(lig_df[col], errors='coerce')
        
        # Get the last row (final time point) for each ligament column
        final_forces = lig_df[lig_def_names].iloc[-1]
        
        # Build summary table
        summary_df = pd.DataFrame({
            'Ligament': final_forces.index,
            'Force_N':  final_forces.values
        })
        
        if verbose:
            print(summary_df.to_string(index=False))
        
        # Write to CSV
        out_path = self.lig_update_dir / f"{self.subject}_Ligament_PeakForces.csv"
        summary_df.to_csv(out_path, index=False)
        
        if verbose:
            print(f"Final ligament forces written to: {out_path}")
        
    
        
        display_df = summary_df.copy()
        mask = display_df['Ligament'] == 'OBJ_SummedForceErrors'
        display_df = pd.concat([display_df[~mask], display_df[mask]], ignore_index=True)
        display_df['Force_N'] = display_df['Force_N'].map('{:20.4f}'.format)

        print(f"\nLigament Peak Forces — {self.subject}")
        print("-" * 45)
        print(display_df.to_string(index=False))
        print("-" * 45)
        
        return summary_df