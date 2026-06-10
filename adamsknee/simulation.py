import pandas as pd
from pathlib import Path

class SimTestMixin:
    def simLaxity(self, test = 'CompPostAnt', compforce=10, pclcond='rPCL', flex_angle = 0, verbose=False):
        self.pclcond = pclcond
        self.flex_angle = flex_angle
        self.test = test
        self.compforce = 10  # This can be parameterized if needed
        cmd_file = self.cmd_path("createSimTest")
        if verbose:
            print(f"Creating simulation CMD file at: {cmd_file}")
        with open(cmd_file, "w", encoding="utf-8") as fid:
            self._write_simLaxity(fid, verbose=verbose)
        self.run_adams(cmd_file)

    def _write_simLaxity(self, fid, verbose=False):
        # Order of naming:
            # subject
            # Test
            # compforce
            # PCL condition
            # Flexion angle
        modelName = self.subject
        # Read binary model
        if self.flex_angle == 0.0:
            input_bin_name = f"{self.subject}_C5_postOpt"
        else:
            input_bin_name = f"{self.subject}_PassiveFLexion_{self.compforce}_{self.pclcond}_{int(self.flex_angle)}d"
        fid.write(f'file bin read file="{self.bin_dir}/{input_bin_name}.bin" \n!\n!\n')


        if self.flex_angle == 90 or self.flex_angle == 120:
            fid.write(f'model delete model_name=.{modelName}\n')
            fid.write(f'entity modify entity = .{modelName}_PassFlex_{self.flex_angle} new = .{modelName}\n!\n')
            # reset
            fid.write(f'marker modify marker_name = .{modelName}.Fem.FixFemToGround  &\n')
            fid.write('orientation = 0.0, 0, 0.0  &\n &\n &\n')
            fid.write(f'relative_to = .{modelName}\n\n')
            fid.write(f'marker modify marker_name = .{modelName}.ground.FixFemToGround  &\n')
            fid.write('orientation = 0.0, 0, 0.0  &\n &\n &\n')
            fid.write(f'relative_to = .{modelName}\n\n')

        
        PCL_bundle_ordered = ['PCL_5', 'PCL_6', 'PCL_7', 'PCL_4', 'PCL_3', 'PCL_2', 'PCL_1']
        if self.pclcond == 'cPCL':
            for iLigament in PCL_bundle_ordered:
                fid.write(f'entity attr entity_name=.{modelName}.{iLigament} active=off dependents_active=off\r\n')
            fid.write(f'entity attr entity_name=.{modelName}.Constraint1_PCLForce active=off dependents_active=off\r\n')
            fid.write(f'entity attr entity_name=.{modelName}.Constraint2_PCLForce active=off dependents_active=off\r\n')
            fid.write(f'entity attr entity_name=.{modelName}.OBJ_SummedForceErrors active=off dependents_active=off\r\n')
            fid.write('\r\n')
        
        # deactivate/activate conditions
        fid.write(f'entity attr entity_name=.{modelName}.FixFlex_0d active=on dependents_active=on\n')
        fid.write(f'entity attr entity_name=.{modelName}.AxialConstraint active=off dependents_active=off\n')
        fid.write(f'entity attr entity_name=.{modelName}.FemFlexion active=off dependents_active=off\n')
        fid.write(f'entity attr entity_name=.{modelName}.FixFemToGround active=on dependents_active=on\n')
        fid.write(f'entity attr entity_name=.{modelName}.FemFlexionRotation active=off dependents_active=off\n')
        fid.write(f'entity attr entity_name=.{modelName}.ForceCD active=on dependents_active=on\n')
        fid.write(f'entity attr entity_name=.{modelName}.MomVV_0d active=off dependents_active=off\n')
        fid.write(f'entity attr entity_name=.{modelName}.Mom_IE active=off dependents_active=off\n')
        fid.write(f'entity attr entity_name=.{modelName}.ForceAP_0d active=on dependents_active=on\n!\n')

        if self.test == 'CompAnt':
            fid.write('force modify direct single_component_force  &\n')
            fid.write(f'single_component_force = .{modelName}.ForceAP_0d &\n')
            fid.write('function = "-30*step(time, 1, 0, 3, 1)"\n')
            totalTime = 3.0
        elif self.test == 'CompPost':
            fid.write('force modify direct single_component_force  &\n')
            fid.write(f'single_component_force = .{modelName}.ForceAP_0d &\n')
            fid.write('function = "30*step(time, 1, 0, 3, 1)"\n')
            totalTime = 3.0
        elif self.test == 'CompPostAnt':
            fid.write('force modify direct single_component_force  &\n')
            fid.write(f'single_component_force = .{modelName}.ForceAP_0d &\n')
            fid.write('function = "30*step(time, 3, 0, 5, 1) - 60*step(time, 5, 0, 7, 1)"\n')
            totalTime = 7.0
        elif self.test == 'Sag':
            fid.write('force modify direct single_component_force  &\n')
            fid.write(f'single_component_force = .{modelName}.ForceAP_0d &\n')
            fid.write('function = "30*step(time, 1, 0, 3, 1)"\n')
            totalTime = 3.0
        elif self.test == 'CompValVar':
            fid.write(f'entity attr entity_name=.{modelName}.ForceAP_0d active=off dependents_active=off\n!\n')
            fid.write(f'entity attr entity_name=.{modelName}.MomVV_0d active=on dependents_active=on\n')
            fid.write('force modify direct single_component_force  &\n')
            fid.write(f'single_component_force = .{modelName}.MomVV_0d &\n')
            fid.write('function = "8000*step(time,3,0,5.5,1) -8000*step(time,7.5,0,10,1) - 8000*step(time,10,0,12.5,1)"\n')
            # fid.write('function = "-16000*step(time,7.5,0,10,1)"\n')
            totalTime = 14.0
            

        fid.write('! ----- Activating the Ligament Force Graphics ----- !\r\n!\r\n')

        if self.pclcond == 'cPCL':
            ligs = [
                'LCL', 'FFL', 'ALL', 'OPL_PL', 'OPL_DL',
                'POL_A', 'POL_C', 'POL_P',
                'PLC_L', 'PLC_C', 'PLC_M',
                'PMC_L', 'PMC_C', 'PMC_M',
                'sMCL_WrapProx_A', 'sMCL_WrapProx_P', 'sMCL_WrapProx_C'
            ]
        elif self.pclcond == 'cPCLAL':
            ligs = [
                'PCL_1', 'PCL_2', 'PCL_3', 'PCL_4',
                'LCL', 'FFL', 'ALL', 'OPL_PL', 'OPL_DL',
                'POL_A', 'POL_C', 'POL_P',
                'PLC_L', 'PLC_C', 'PLC_M',
                'PMC_L', 'PMC_C', 'PMC_M',
                'sMCL_WrapProx_A', 'sMCL_WrapProx_P', 'sMCL_WrapProx_C'
            ]
        else:
            ligs = [
                'PCL_1', 'PCL_2', 'PCL_3', 'PCL_4',
                'PCL_5', 'PCL_6', 'PCL_7',
                'LCL', 'FFL', 'ALL', 'OPL_PL', 'OPL_DL',
                'POL_A', 'POL_C', 'POL_P',
                'PLC_L', 'PLC_C', 'PLC_M',
                'PMC_L', 'PMC_C', 'PMC_M',
                'sMCL_WrapProx_A', 'sMCL_WrapProx_P', 'sMCL_WrapProx_C'
            ]

        for lig in ligs:
            fid.write(f'mdi graphic_force object = .{modelName}.{lig} type = 2 \n')
        
        # Simulation
        fid.write('! ----- Forward Simulation ----- !\n')
        if verbose:
            print(f"Total simulation time: {totalTime:.2f} seconds")
        fid.write(f'simulation single_run transient type=dynamic initial_static=no duration={totalTime} step_size=0.01\n!\n')
        # fid.write(f'simulation single_run transient type=dynamic initial_static=no duration={totalTime} number_of_steps = 2000\n!\n')

        output_name = f"{self.subject}_{self.test}_{self.compforce}_{self.pclcond}_{int(self.flex_angle)}d"

        # Export Analysis
        xvar = ['Force_FFL', 'Force_LCL', 'Force_OPL_PL', 'OBJ_SummedForceErrors', 'TotalForce_PCL_PM' , 'TotalForce_PCL_AL',\
                'TotalForce_PLC', 'TotalForce_PMC', 'TotalForce_POL', 'TotalForce_sMCL_WrapDist', 'TotalForce_sMCL_WrapProx',\
                'Force_ALL', 'Alpha', 'Beta', 'Gamma', 'q1', 'q2', 'q3', 'Applied_ForceCD'
                ]
        fid.write('! ----- Export Analysis ----- !\n')
        fid.write('xy_plot template modify plot=.plot_1 auto_title=yes auto_subtitle=yes auto_date=yes auto_analysis_name=yes table=no\n')
        fid.write('xy_plot template clear plot=.plot_1\n')
        for kvar in range(len(xvar)):
            fid.write(f'xy_plot curve create curve=.plot_1.curve_{kvar+1} \
                      create_page=no calculate_axis_limits=no dmeasure=.{modelName}.{xvar[kvar]} \
                      run=.{modelName}.Last_Run auto_axis=UNITS\n')
        kvar += 1

        if self.case == 'PS':
                bodyobjects = [
                'FemComp',
                'Tib',
                'InsertBase',
                'InsertMed',
                'InsertLat',
                'InsertPost'
            ]  # These names should match the names of STL Files in the Data_Raw\Geometries folder
        else:
            bodyobjects = [
            'FemComp',
            'Tib',
            'Insert'
            ]  # These names should match the names of STL Files in the Data_Raw\Geometries folder
        components = ['_XFORM.X', '_XFORM.Y', '_XFORM.Z', '_XFORM.PSI', '_XFORM.THETA', '_XFORM.PHI']
        newmeasures = []
        for ipart in bodyobjects:
            for jcomp in components:
                newmeasures.append(ipart + jcomp)
        for measure in newmeasures:
            kvar += 1
            fid.write(f'xy_plot curve create curve=.plot_1.curve_{kvar} create_page=no calculate_axis_limits=no ddata=.{modelName}.Last_Run.{measure} run=.{modelName}.Last_Run auto_axis=UNITS \n')
        fid.write('xy_plot template calculate_axis_limits plot_name=.plot_1\n')
        fid.write('file table write  &\n')
        fid.write(f'file_name = "{self.output_dir}/{output_name}" &\n')
        fid.write('plot_name = .plot_1 &\n')
        fid.write('format = spreadsheet\n!\n')

        fid.write('! ----- Save the binary model ----- !\n')
        # output_bin_name = f"{self.subject}_10_{self.test}_{self.pclcond}_{int(self.flex_angle)}d"
        fid.write(f'file bin write file="{self.bin_dir}/{output_name}.bin" \n!\n!\n')
        fid.write('!\n')
        
    
    
    
    
    
    
    # Simulation of Passive Flexion
    def simFlexion(self, pclcond='rPCL', verbose=False, run_adams=True):
        self.pclcond = pclcond
        cmd_file = self.cmd_path("createSimFlexion")
        if verbose:
            print(f"Creating simulation CMD file at: {cmd_file}")
        with open(cmd_file, "w", encoding="utf-8") as fid:
            self._write_simFlexion(fid, verbose=verbose)
        if run_adams:
            self.run_adams(cmd_file)
        
    def _write_simFlexion(self, fid, verbose=False):
        large_angles = [90, 120]
        angles = [30, 45]
        modelName = self.subject
        # Read binary model
        for kk, flex_angle in enumerate(angles):
            if kk == 0:
                dangle = flex_angle - 0
                input_bin_name = f"{self.subject}_C5_postOpt"
            else:
                dangle = flex_angle - angles[kk-1]
                input_bin_name = f"{self.subject}_PassiveFLexion_10_{self.pclcond}_{int(angles[kk-1])}d"
            
            fid.write(f'file bin read file="{self.bin_dir}/{input_bin_name}.bin" \n!\n!\n')
            
            # deactivate/activate motions
            fid.write(f'entity attr entity_name=.{modelName}.FixFlex_0d active=on dependents_active=on\n')
            fid.write(f'entity attr entity_name=.{modelName}.AxialConstraint active=off dependents_active=off\n')
            fid.write(f'entity attr entity_name=.{modelName}.FemFlexion active=on dependents_active=on\n')
            fid.write(f'entity attr entity_name=.{modelName}.FixFemToGround active=off dependents_active=off\n')
            fid.write(f'entity attr entity_name=.{modelName}.FemFlexionRotation active=on dependents_active=on\n')
            # deactivate/activate forces
            fid.write(f'entity attr entity_name=.{modelName}.ForceCD active=on dependents_active=on\n')
            fid.write(f'entity attr entity_name=.{modelName}.MomVV_0d active=off dependents_active=off\n')
            fid.write(f'entity attr entity_name=.{modelName}.Mom_IE active=off dependents_active=off\n')
            fid.write(f'entity attr entity_name=.{modelName}.ForceAP_0d active=off dependents_active=off\n!\n')
            
            #update axial load
            fid.write('force modify direct single_component_force  &\n')
            fid.write(f'single_component_force = .{modelName}.ForceCD &\n')
            fid.write(f'function = "10*step(time, 0.5, 0, 1, 1)"\n')
            fid.write('!\n')
            fid.write('!\n')
            
            PCL_bundle_ordered = ['PCL_5', 'PCL_6', 'PCL_7', 'PCL_4', 'PCL_3', 'PCL_2', 'PCL_1']
            if self.pclcond == 'cPCL':
                for iLigament in PCL_bundle_ordered:
                    fid.write(f'entity attr entity_name=.{modelName}.{iLigament} active=off dependents_active=off\r\n')
                fid.write(f'entity attr entity_name=.{modelName}.Constraint1_PCLForce active=off dependents_active=off\r\n')
                fid.write(f'entity attr entity_name=.{modelName}.Constraint2_PCLForce active=off dependents_active=off\r\n')
                fid.write(f'entity attr entity_name=.{modelName}.OBJ_SummedForceErrors active=off dependents_active=off\r\n')
            elif self.pclcond == 'cPCLAL':
                for iLigament in ['PCL_5', 'PCL_6', 'PCL_7']:
                    fid.write(f'entity attr entity_name=.{modelName}.{iLigament} active=off dependents_active=off\r\n')
                fid.write(f'entity attr entity_name=.{modelName}.Constraint1_PCLForce active=off dependents_active=off\r\n')
                fid.write(f'entity attr entity_name=.{modelName}.Constraint2_PCLForce active=off dependents_active=off\r\n')
                fid.write(f'entity attr entity_name=.{modelName}.OBJ_SummedForceErrors active=off dependents_active=off\r\n')
            fid.write('!\n')
            
            readligs = ['PCL_PM', 'PCL_AL', 'sMCL_Prox', 'sMCL_Dist', 'LCL', 'POL', 'FFL', 'OPL', 'PMC', 'PLC']
            # change this to where the optimized file is

            for lig in readligs:
                if lig == 'PCL_AL':
                    l0percent = 1.1
                    fid.write(f'variable modify variable_name = .{self.subject}.Percent_L0_{lig} real ={l0percent}\n')
            
            # update flexion motion function
            fid.write(f'constraint modify motion motion_name = .{modelName}.FemFlexionRotation &\n')
            fid.write(f'function = "Step(time, 1, {angles[kk] - dangle}d, {dangle + 1}, {angles[kk]}d)"\n!\n')

            # run simulation
            duration = 1 + dangle
            step_size = 0.01
            fid.write(f'simulation single_run transient type=dynamic initial_static=no duration={duration} step_size={step_size}\n!\n')

            last_step = duration / step_size + 1

            # save at last frame
            fid.write(f'model copy new_model_name = .{modelName}_PassFlex &\n')
            fid.write(f'analysis = (.{modelName}.Last_Run) &\n')
            fid.write(f'frame_number = ({int(last_step)}) &\n')
            fid.write('view_name = all &\n')
            fid.write('include_contact_steps="no"\n!\n')
            fid.write(f'model delete model_name=.{modelName}\n')
            fid.write(f'model display model_name=.{modelName}_PassFlex view_name=.gui.main.*\n')
            fid.write(f'entity modify entity = .{modelName}_PassFlex new = .{modelName}\n!\n')

            # reset
            fid.write(f'marker modify marker_name = .{modelName}.Fem.FixFemToGround  &\n')
            fid.write('orientation = 0.0, 0, 0.0  &\n &\n &\n')
            fid.write(f'relative_to = .{modelName}\n')

            output_bin_name = f"{self.subject}_PassiveFLexion_10_{self.pclcond}_{int(angles[kk])}d"
            fid.write('! ----- Save the binary model ----- !\n')
            fid.write(f'file bin write file="{self.bin_dir}/{output_bin_name}.bin" \n!\n!\n')
            fid.write('!\n')
        
        # fid.write('quit conf=no')
        
        # -------------------------------------------------------
        # -------------------------------------------------------
        # Large Angles
        # -------------------------------------------------------
        # -------------------------------------------------------

        
        compforces = [10, 500]
        for _, compforce in enumerate(compforces):
            
            for kk, largeflex_angle in enumerate(large_angles):
                fid.write('! ----- Passive Flexion ----- !\r\n!\r\n')
                input_bin_name = f"{self.subject}_C5_postOpt"
                fid.write(f'file bin read file="{self.bin_dir}/{input_bin_name}.bin" \n!\n!\n')
                
                # deactivate/activate motions
                fid.write(f'entity attr entity_name=.{modelName}.FixFlex_0d active=on dependents_active=on\n')
                fid.write(f'entity attr entity_name=.{modelName}.AxialConstraint active=off dependents_active=off\n')
                fid.write(f'entity attr entity_name=.{modelName}.FemFlexion active=on dependents_active=on\n')
                fid.write(f'entity attr entity_name=.{modelName}.FixFemToGround active=off dependents_active=off\n')
                fid.write(f'entity attr entity_name=.{modelName}.FemFlexionRotation active=on dependents_active=on\n')
                # deactivate/activate forces
                fid.write(f'entity attr entity_name=.{modelName}.ForceCD active=on dependents_active=on\n')
                fid.write(f'entity attr entity_name=.{modelName}.MomVV_0d active=off dependents_active=off\n')
                fid.write(f'entity attr entity_name=.{modelName}.Mom_IE active=off dependents_active=off\n')
                fid.write(f'entity attr entity_name=.{modelName}.ForceAP_0d active=off dependents_active=off\n!\n')
                
                #update axial load
                fid.write('force modify direct single_component_force  &\n')
                fid.write(f'single_component_force = .{modelName}.ForceCD &\n')
                fid.write(f'function = "{compforce}*step(time, 0.5, 0, 1, 1)"\n')
                fid.write('!\n')
                fid.write('!\n')
                
                PCL_bundle_ordered = ['PCL_5', 'PCL_6', 'PCL_7', 'PCL_4', 'PCL_3', 'PCL_2', 'PCL_1']
                if self.pclcond == 'cPCL':
                    for iLigament in PCL_bundle_ordered:
                        fid.write(f'entity attr entity_name=.{modelName}.{iLigament} active=off dependents_active=off\r\n')
                    fid.write(f'entity attr entity_name=.{modelName}.Constraint1_PCLForce active=off dependents_active=off\r\n')
                    fid.write(f'entity attr entity_name=.{modelName}.Constraint2_PCLForce active=off dependents_active=off\r\n')
                    fid.write(f'entity attr entity_name=.{modelName}.OBJ_SummedForceErrors active=off dependents_active=off\r\n')
                elif self.pclcond == 'cPCLAL':
                    for iLigament in ['PCL_5', 'PCL_6', 'PCL_7']:
                        fid.write(f'entity attr entity_name=.{modelName}.{iLigament} active=off dependents_active=off\r\n')
                    fid.write(f'entity attr entity_name=.{modelName}.Constraint1_PCLForce active=off dependents_active=off\r\n')
                    fid.write(f'entity attr entity_name=.{modelName}.Constraint2_PCLForce active=off dependents_active=off\r\n')
                    fid.write(f'entity attr entity_name=.{modelName}.OBJ_SummedForceErrors active=off dependents_active=off\r\n')
                fid.write('!\n')
                
                readligs = ['PCL_PM', 'PCL_AL', 'sMCL_Prox', 'sMCL_Dist', 'LCL', 'POL', 'FFL', 'OPL', 'PMC', 'PLC']
                # change this to where the optimized file is
                for lig in readligs:
                    if lig == 'PCL_AL':
                        l0percent = 1.1
                        fid.write(f'variable modify variable_name = .{self.subject}.Percent_L0_{lig} real ={l0percent}\n')
                
                # update flexion motion function
                fid.write(f'constraint modify motion motion_name = .{modelName}.FemFlexionRotation &\n')
                fid.write(f'function = "Step(time, 1, 0d, {largeflex_angle + 1}, {largeflex_angle}d)"\n!\n')
                
                fid.write('! ----- Activating the Ligament Force Graphics ----- !\r\n!\r\n')

                if self.pclcond == 'cPCL':
                    ligs = [
                        'LCL', 'FFL', 'ALL', 'OPL_PL', 'OPL_DL',
                        'POL_A', 'POL_C', 'POL_P',
                        'PLC_L', 'PLC_C', 'PLC_M',
                        'PMC_L', 'PMC_C', 'PMC_M',
                        'sMCL_WrapProx_A', 'sMCL_WrapProx_P', 'sMCL_WrapProx_C'
                    ]
                elif self.pclcond == 'cPCLAL':
                    ligs = [
                        'PCL_1', 'PCL_2', 'PCL_3', 'PCL_4',
                        'LCL', 'FFL', 'ALL', 'OPL_PL', 'OPL_DL',
                        'POL_A', 'POL_C', 'POL_P',
                        'PLC_L', 'PLC_C', 'PLC_M',
                        'PMC_L', 'PMC_C', 'PMC_M',
                        'sMCL_WrapProx_A', 'sMCL_WrapProx_P', 'sMCL_WrapProx_C'
                    ]
                else:
                    ligs = [
                        'PCL_1', 'PCL_2', 'PCL_3', 'PCL_4',
                        'PCL_5', 'PCL_6', 'PCL_7',
                        'LCL', 'FFL', 'ALL', 'OPL_PL', 'OPL_DL',
                        'POL_A', 'POL_C', 'POL_P',
                        'PLC_L', 'PLC_C', 'PLC_M',
                        'PMC_L', 'PMC_C', 'PMC_M',
                        'sMCL_WrapProx_A', 'sMCL_WrapProx_P', 'sMCL_WrapProx_C'
                    ]

                for lig in ligs:
                    fid.write(f'mdi graphic_force object = .{modelName}.{lig} type = 2 \n')
                    
                # run simulation
                duration = 1 + largeflex_angle
                step_size = 0.01
                fid.write(f'simulation single_run transient type=dynamic initial_static=no duration={duration} step_size={step_size}\n!\n')
                
                output_name = f"{self.subject}_PassiveFlexion_{int(compforce)}_{self.pclcond}_{int(largeflex_angle)}d"
                # Export Analysis
                xvar = ['Force_FFL', 'Force_LCL', 'Force_OPL_PL', 'OBJ_SummedForceErrors', 'TotalForce_PCL_PM' , 'TotalForce_PCL_AL',\
                        'TotalForce_PLC', 'TotalForce_PMC', 'TotalForce_POL', 'TotalForce_sMCL_WrapDist', 'TotalForce_sMCL_WrapProx',\
                        'Force_ALL', 'Alpha', 'Beta', 'Gamma', 'q1', 'q2', 'q3', 'Applied_ForceCD'
                        ]
                fid.write('! ----- Export Analysis ----- !\n')
                fid.write('xy_plot template modify plot=.plot_1 auto_title=yes auto_subtitle=yes auto_date=yes auto_analysis_name=yes table=no\n')
                fid.write('xy_plot template clear plot=.plot_1\n')
                for kvar in range(len(xvar)):
                    fid.write(f'xy_plot curve create curve=.plot_1.curve_{kvar+1} \
                            create_page=no calculate_axis_limits=no dmeasure=.{modelName}.{xvar[kvar]} \
                            run=.{modelName}.Last_Run auto_axis=UNITS\n')
                kvar += 1
                if self.case == 'PS':
                    bodyobjects = [
                    'FemComp',
                    'Tib',
                    'InsertBase',
                    'InsertMed',
                    'InsertLat',
                    'InsertPost'
                    ]  # These names should match the names of STL Files in the Data_Raw\Geometries folder
                else:
                    bodyobjects = [
                        'FemComp',
                        'Tib',
                        'Insert'
                    ]  # These names should match the names of STL Files in the Data_Raw\Geometries folder
                components = ['_XFORM.X', '_XFORM.Y', '_XFORM.Z', '_XFORM.PSI', '_XFORM.THETA', '_XFORM.PHI']
                newmeasures = []
                for ipart in bodyobjects:
                    for jcomp in components:
                        newmeasures.append(ipart + jcomp)
                for measure in newmeasures:
                    kvar += 1
                    fid.write(f'xy_plot curve create curve=.plot_1.curve_{kvar} create_page=no calculate_axis_limits=no ddata=.{modelName}.Last_Run.{measure} run=.{modelName}.Last_Run auto_axis=UNITS \n')
                fid.write('xy_plot template calculate_axis_limits plot_name=.plot_1\n')
                fid.write('file table write  &\n')
                fid.write(f'file_name = "{self.output_dir}/{output_name}" &\n')
                fid.write('plot_name = .plot_1 &\n')
                fid.write('format = spreadsheet\n!\n')

                last_step = duration / step_size + 1

                # save at last frame
                fid.write(f'model copy new_model_name = .{modelName}_PassFlex_{largeflex_angle} &\n')
                fid.write(f'analysis = (.{modelName}.Last_Run) &\n')
                fid.write(f'frame_number = ({int(last_step)}) &\n')
                fid.write('view_name = all &\n')
                fid.write('include_contact_steps="no"\n!\n')
                # fid.write(f'model delete model_name=.{modelName}\n')
                fid.write(f'model display model_name=.{modelName}_PassFlex_{largeflex_angle} view_name=.gui.main.*\n')
                # fid.write(f'entity modify entity = .{modelName}_PassFlex_{large_angle} new = .{modelName}\n!\n')

                # reset
                fid.write(f'marker modify marker_name = .{modelName}_PassFlex_{largeflex_angle}.Fem.FixFemToGround  &\n')
                fid.write('orientation = 90.0, 0, 270.0  &\n &\n &\n')
                fid.write(f'relative_to = .{modelName}_PassFlex_{largeflex_angle}\n\n')
    
    
                fid.write('! ----- Save the binary model ----- !\n')
                # output_bin_name = f"{self.subject}_10_{self.test}_{self.pclcond}_{int(self.flex_angle)}d"
                fid.write(f'file bin write file="{self.bin_dir}/{output_name}.bin" \n!\n!\n')
                fid.write('!\n')
        # fid.write('quit conf=no')
    
    
    ########################################################
    # Contact Processing
    ########################################################
    
    def process_contact(self, test='PassiveFlexion', compforce=500, pclcond='rPCL', flex_angle=90, run_adams= True, verbose=False):
        """
        Process contact data for the simulation of passive flexion.
        """
        # Set instance attributes, just like simLaxity and simFlexion do
        self.test = test
        self.compforce = compforce
        self.pclcond = pclcond
        self.flex_angle = flex_angle

        self._createcontactfolder()
        cmd_file = self.cmd_path('saveContact')
        if verbose:
            print(f"Creating simulation CMD file at: {cmd_file}")
        with open(cmd_file, "w", encoding="utf-8") as fid:
            self._write_contact(fid, verbose=verbose)  # also removed the erroneous extra `self`
        if run_adams:
            self.run_adams(cmd_file)

    def _createcontactfolder(self):
        contact_folder = self.contact_dir / f"{self.subject}_{self.test}_{self.compforce}_{self.pclcond}_{self.flex_angle}d"
        if contact_folder.exists():
            print(f"Contact folder already exists: {contact_folder}")
        else:
            try:
                contact_folder.mkdir(parents=False, exist_ok=True)
                print(f"Created contact folder: {contact_folder}")
            except Exception as e:
                print(f"Failed to create contact folder {contact_folder}: {e}")
        self.contact_folder = contact_folder

    def _write_contact(self, fid, verbose=False):
        
        output_name = (f"{self.subject}_{self.test}_{self.compforce}_"
                       f"{self.pclcond}_{int(self.flex_angle)}d")
        motion_output = pd.read_csv(
            self.output_dir / f"{output_name}.tab", sep='\t', skiprows=1)
        motion_output.columns = [c.strip() for c in motion_output.columns]
        time       = motion_output['Time'].values
        num_frames = len(time)
        
        num_incidents_file = self.contact_folder / 'num_incidents'
        contact_incidents_files = self.contact_folder / '*CONTACT_INCIDENT*'

        # Clean up the folder
        for leftoverfile in (num_incidents_file, contact_incidents_files):
            p = Path(leftoverfile) 
            if p.exists():
                p.unlink()
                
        

        with open(num_incidents_file, "w", encoding="utf-8") as fnum_incidents:
            fnum_incidents.write(str(num_frames))

        input_bin_name = f"{self.subject}_{self.test}_{int(self.compforce)}_{self.pclcond}_{int(self.flex_angle)}d"
        fid.write(f'file bin read file="{self.bin_dir}/{input_bin_name}.bin" \n!\n!\n')
        fid.write(f'interface dialog undisplay dialog=.gui.info_window\n')
        fid.write(f'!\n')

        modelName = self.subject
        for inc in range(20,num_frames,100):
            fid.write(f'list_info entity entity_name = .{modelName}.last_run.contact_incidents.incident_{str(int(inc))} brief=off &\n')
            fid.write(f'file_name = "{self.contact_folder}/CONTACT_INCIDENT_{str(int(inc))}" \n')
            fid.write('!\n')
        

