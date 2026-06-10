from registrations import (
    read_tfm_file,
)

class ImportGeometriesMixin:
    def import_geometries(self, verbose=False, run_adams=True):
        cmd_file = self.cmd_path("step1_import_geometries")
        self.clean_cmd_dir()
        if verbose:
            print(f"Creating simulation CMD file at: {cmd_file}")
        with open(cmd_file, "w", encoding="utf-8") as fid:
            self._write_import_geometries(fid, verbose=verbose)
        if run_adams:
            self.run_adams(cmd_file)

    def _write_import_geometries(self, fid, verbose=False):
        
        model_name = f"{self.subject}"            
        _, euler_xyz_Fem =read_tfm_file( self.transforms_dir / "RI.tfm")
        _, euler_xyz_Tib =read_tfm_file( self.transforms_dir / "RI.tfm")
        
        part_names       = ['Fem', 'FemComp', 'Insert', 'Tray', 'Tib', 'Fib']
        parasolid_names  = ['Fem.SOLID1', 'FemComp.SOLID2', 'Insert.SOLID3',
                            'Tray.SOLID4', 'Tib.SOLID5', 'Fib.SOLID6']
        parasolid_colors = ['YELLOW', 'YELLOW', 'CYAN', 'WHITE', 'WHITE', 'WHITE']
        
        
        # ----- 1. Units and Coordinates -----
        fid.write('! ----- 1. Units and Coordinates ----- !\n')
        fid.write('! Default Units for Model\n')
        fid.write('defaults units &\n')
        fid.write('    length = mm &\n')
        fid.write('    angle = deg &\n')
        fid.write('    force = newton &\n')
        fid.write('    mass = kg &\n')
        fid.write('    time = sec\n')
        fid.write('!\n')
 
        fid.write('defaults units &\n')
        fid.write('    coordinate_system_type = cartesian &\n')
        fid.write('    orientation_type = body123\n')
        fid.write('!\n')
 
        fid.write(f'model create model="{model_name}"\n')
        fid.write('!\n')
 
        fid.write(f'defaults coordinate_system  default_coordinate_system = .{model_name}.ground\n')
        fid.write('!\n')
        
        input_bin_file = f'{self.subject}'
        fid.write(f'file bin write file="{self.bin_dir}/{input_bin_file}.bin" \n!\n')
        
        # ----- 2. Create Parts -----
        fid.write('! ----- 2. Create Parts ----- !\n!\n')
        for part in part_names:
            fid.write(f'part create rigid_body name_and_position  part_name = .{model_name}.{part}  &\n')
            fid.write('    ground_part = no  &\n')
            fid.write('    planar = no  &\n')
            fid.write('    planar_axes = xy  &\n')
            fid.write('    orientation = 0.0, 0.0, 0.0  \n')
            fid.write('!\n')
 
        # ----- 3. Import ASCII STL Files -----
        fid.write('! ----- 3. Import ASCII STL Files ----- !\n!\n')
        for part in part_names:
            fid.write('file stereo read  &\n')
            fid.write(f'   file_name = "{self.geom_dir}/{part}.stl"  &\n')
            fid.write(f'   part_name = .{model_name}.{part}  &\n')
            fid.write('   scale = 1.0  &\n')
            fid.write('   orientation = 0.0, 0.0, 0.0  &\n')
            fid.write(f'   relative_to = {model_name}\n')
            fid.write('!\n')
 
        # ----- 4. Export Binary Parasolids -----
        fid.write('! ----- 4. Export Binary Parasolids ----- !\n!\n')
        for part in part_names:
            fid.write('file parasolid write &\n')
            fid.write(f'   file_name = "{self.geom_dir}/{part}"  &\n')
            fid.write('   type = binary  &\n')
            fid.write(f'   part_name = .{model_name}.{part}\n')
            fid.write('!\n')
 
        # ----- 5. Delete the STL Geometries -----
        fid.write('! ----- 5. Delete the STL Geometries ----- !\n!\n')
        for part in part_names:
            fid.write(f'group modify group=SELECT_LIST obj=.{model_name}.{part}.wrap expand_groups=no \n')
            fid.write('mdi delete_macro \n')
            fid.write('!\n')
 
        # ----- 6. Import Parasolids -----
        fid.write('! ----- 6. Import Parasolids ----- !\n!\n')
        for part in part_names:
            fid.write('file parasolid read &\n')
            fid.write(f'   file_name = "{self.geom_dir}/{part}.xmt_bin"  &\n')
            fid.write('   type = BINARY  &\n')
            fid.write(f'   part_name = .{model_name}.{part} &\n')
            fid.write('  orientation = 0.0, 0.0, 0.0 &\n')
            fid.write('   explode_assemblies = no\n')
 
        # ----- 7. Move Geometries to Build Position -----
        fid.write('! ----- 7. Move Geometries to Build Position ----- !\n!\n')
        tib_parts = {'Tib', 'Tray', 'Insert', 'Fib'}
        fem_parts = {'Fem', 'FemComp'}
        for part in part_names:
            if part in tib_parts:
                xyz123 = euler_xyz_Tib
            elif part in fem_parts:
                xyz123 = euler_xyz_Fem
            fid.write(f'move object geometry = {part}.wrap &\n')
            fid.write(f'    c1={xyz123[0]:f} &\n')
            fid.write(f'    c2={xyz123[1]:f} &\n')
            fid.write(f'    c3={xyz123[2]:f} &\n')
            fid.write(f'    a1={xyz123[3]:f} &\n')
            fid.write(f'    a2={xyz123[4]:f} &\n')
            fid.write(f'    a3={xyz123[5]:f}  \n')
            fid.write('!\n')
 
        # ----- 9. Assign Unique Names to Geometries -----
        fid.write('! ----- 9. Assign Unique names to Geometries ----- !\n!\n')
        fid.write(f'entity modify entity = .{model_name}.Fem.wrap new = .{model_name}.Fem.SOLID1\n')
        fid.write(f'entity modify entity = .{model_name}.FemComp.wrap new = .{model_name}.FemComp.SOLID2\n')
        fid.write(f'entity modify entity = .{model_name}.Insert.wrap new = .{model_name}.Insert.SOLID3\n')
        fid.write(f'entity modify entity = .{model_name}.Tray.wrap new = .{model_name}.Tray.SOLID4\n')
        fid.write(f'entity modify entity = .{model_name}.Tib.wrap new = .{model_name}.Tib.SOLID5\n')
        fid.write(f'entity modify entity = .{model_name}.Fib.wrap new = .{model_name}.Fib.SOLID6\n')
        fid.write('!\n')
 
        # ----- 10. Adjust Graphics -----
        fid.write('! ----- 10. Adjsut Graphics ----- !\n!\n')
        for ps_name, ps_color in zip(parasolid_names, parasolid_colors):
            fid.write(f'geometry attributes  geometry_name = .{model_name}.{ps_name}  &\n')
            fid.write(f'    color = {ps_color}  &\n')
            fid.write('    visibility = on\n')
            fid.write('!\n')
 
        for part in ('Fem', 'Tib', 'Fib'):
            fid.write(f'part attributes  part_name = .{model_name}.{part} &\n')
            fid.write('    color = WHITE  &\n')
            fid.write('    transparency = 42\n')
        fid.write('!\n')
 
        fid.write('view man mod render=sshaded')
        fid.write('!\n')
        
        output_bin_file = f'{self.subject}_C1'
        fid.write(f'file bin write file="{self.bin_dir}/{output_bin_file}.bin" \n!\n')