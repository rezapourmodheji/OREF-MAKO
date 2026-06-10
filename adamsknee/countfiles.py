from datetime import datetime
import itertools
import shutil


class CountFilesMixin:
    def count_files(self, search_terms=["_C5_postOpt.bin"], numfile_expected=1, check_missing=False, recursive=False):
        print(f"\n=== Processing for {self.subject} {self.case} {self.coltension} ===\n")


        ## Search BIN directory
        for term in search_terms:
            found_terms = []
            if not check_missing:
                print(f"Searching for BIN files containing: {term} \n")
            for file_name in self.bin_dir.rglob(f'*{term}*') if recursive else self.bin_dir.glob(f'*{term}*'):
                stat = file_name.stat()
                size_mb = stat.st_size / (1024 * 1024)
                modified = datetime.fromtimestamp(stat.st_mtime)
                if not check_missing:
                    print(f"Found file: {file_name.name:<30} | "
                          f"size: {size_mb:6.2f} MB | "
                          f"modified: {modified.strftime('%d %B %Y %H:%M')}")
                found_terms.append(file_name)
            if not check_missing:
                print(f"\n Found {len(found_terms)} files containing: {term} \n")
            if len(found_terms) == numfile_expected:
                if not check_missing:
                    print(f"  All expected files found for term: {term} \n")
            else:
                print(f"  !! WARNING !! number of files found: {len(found_terms)}, expected: {numfile_expected} \n")
        

        ## Search OUTPUT directory
        numfile_expected = 8
        for term in search_terms:
            found_terms = []
            if not check_missing:
                print(f"Searching for OUTPUT files containing: {term} \n")
            for file_name in self.output_dir.rglob(f'*{term}*') if recursive else self.output_dir.glob(f'*{term}*'):
                stat = file_name.stat()
                size_mb = stat.st_size / (1024 * 1024)
                modified = datetime.fromtimestamp(stat.st_mtime)
                if not check_missing:
                    print(f"Found file: {file_name.name:<30} | "
                          f"size: {size_mb:6.2f} MB | "
                          f"modified: {modified.strftime('%d %B %Y %H:%M')}")
                found_terms.append(file_name)
            if not check_missing:
                print(f"\n Found {len(found_terms)} files containing: {term} \n")
            if len(found_terms) == numfile_expected:
                if not check_missing:
                    print(f"  All expected files found for term: {term} \n")
            else:
                print(f"  !! WARNING !! number of files found: {len(found_terms)}, expected: {numfile_expected} \n")

    def count_contact(self, search_terms=["CONTACT_INCIDENT_"],
                numfile_expected=5, check_missing=False, recursive=False, **kwargs):
        contactfilesdir = self.contact_dir / f"{self.subject}_{kwargs.get('test','')}_{kwargs.get('compforce','')}_{kwargs.get('pclcond','')}_{kwargs.get('flex_angle','')}d"
        for term in search_terms:
            found_terms = []
            for file_name in contactfilesdir.rglob(f'*{term}*') if recursive else contactfilesdir.glob(f'*{term}*'):
                found_terms.append(file_name)
            if not check_missing:
                print(f"\n Found {len(found_terms)} files containing: {term} \n")
            if len(list(contactfilesdir.rglob('num_incidents'))) > 0:
                if not check_missing: print(f"  num_incidents file found in {contactfilesdir}")
            else:
                print(f"  !! WARNING !! num_incidents file NOT found in {contactfilesdir}")
            if len(found_terms) >= numfile_expected:
                if not check_missing:
                    print(f"  All expected files found for term: {term} \n")
            else:   print(f"  !! WARNING !! number of files found: {len(found_terms)}, expected at least: {numfile_expected} \n")

                
    
    
    
    
    
    def collect_passive_flexion(self, search_term="_sim_PassiveFlexion", folder_name="PassiveFlexion_collected", recursive=False):
        """
        --------------- Pretty Descriptive -----------
        Identifies files containing '_sim_PassiveFlexion' in their name,
        creates a new folder next to where the files are found,
        and moves the identified files into it.
        """
        print(f"\n=== Collecting PassiveFlexion files for {self.subject} ===\n")

        search_dirs = {
            "BIN":    self.bin_dir,
            "OUTPUT": self.output_dir,
        }

        for dir_label, search_dir in search_dirs.items():
            found_files = []

            glob_fn = search_dir.rglob if recursive else search_dir.glob
            for file_path in glob_fn(f'*{search_term}*'):
                stat = file_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                modified = datetime.fromtimestamp(stat.st_mtime)
                print(f"Found [{dir_label}]: {file_path.name:<40} | "
                      f"size: {size_mb:6.2f} MB | "
                      f"modified: {modified.strftime('%d %B %Y %H:%M')}")
                found_files.append(file_path)

            if not found_files:
                print(f"  No PassiveFlexion files found in {dir_label} directory.\n")
                continue

            dest_folder = found_files[0].parent / folder_name
            dest_folder.mkdir(exist_ok=True)
            print(f"\n  Destination folder: {dest_folder}")

            moved, skipped = 0, 0
            for file_path in found_files:
                dest_path = dest_folder / file_path.name
                if dest_path.exists():
                    print(f"  Skipped (already exists): {file_path.name}")
                    skipped += 1
                else:
                    shutil.move(file_path, dest_path)  
                    print(f"  Moved: {file_path.name}") 
                    moved += 1

            print(f"\n  [{dir_label}] Done — {moved} file(s) moved, {skipped} skipped.\n")
    
    def rename_passive_flexion(self, compforces, pclconds, flex_angles, recursive=True):
        """
        Identifies files containing '_sim_PassiveFlexion_' in their name and renames them
        by removing the '_sim' prefix, in place (no folder created).
        """
        print(f"\n=== Renaming PassiveFlexion files for {self.subject} ===\n")

        search_dirs = {"BIN": self.bin_dir}

        for dir_label, search_dir in search_dirs.items():
            print(f"--- [{dir_label}] directory ---\n")
            total_renamed, total_skipped = 0, 0

            for compforce, pclcond, flex_angle in itertools.product(compforces, pclconds, flex_angles):
                old_term = f"{self.subject}_sim_PassiveFlexion_{int(compforce)}_{pclcond}_{int(flex_angle)}d"
                new_term = f"{self.subject}_PassiveFlexion_{int(compforce)}_{pclcond}_{int(flex_angle)}d"

                glob_fn = search_dir.rglob if recursive else search_dir.glob
                found_files = list(glob_fn(f'*{old_term}*'))

                if not found_files:
                    print(f"  No files found for: {old_term}")
                    continue

                print(f"  Found {len(found_files)} file(s) for: {old_term}")
                for file_path in found_files:
                    stat = file_path.stat()
                    print(f"    {file_path.name:<50} | "
                        f"{stat.st_size / (1024*1024):6.2f} MB | "
                        f"{datetime.fromtimestamp(stat.st_mtime).strftime('%d %B %Y %H:%M')}")

                for file_path in found_files:
                    new_name = file_path.name.replace(old_term, new_term)
                    dest_path = file_path.parent / new_name
                    if dest_path.exists():
                        print(f"  Skipped (already exists): {new_name}")
                        total_skipped += 1
                    else:
                        file_path.rename(dest_path)
                        print(f"  Renamed: {file_path.name}  -->  {new_name}\n")
                        total_renamed += 1

            print(f"\n  [{dir_label}] Done — {total_renamed} file(s) renamed, {total_skipped} skipped.\n")
    
    
    
    
    def collect_passive_flexion_output(self, search_term="ForcePCL", folder_name="PassiveFlexion_collected", recursive=False):
        """
        Identifies files containing '_sim_PassiveFlexion' in their name,
        creates a new folder next to where the files are found,
        and moves the identified files into it.
        """
        print(f"\n=== Collecting PassiveFlexion files for {self.subject} ===\n")

        search_dirs = {
            "BIN":    self.bin_dir,
            "OUTPUT": self.output_dir,
        }

        for dir_label, search_dir in search_dirs.items():
            found_files = []

            glob_fn = search_dir.rglob if recursive else search_dir.glob
            for file_path in glob_fn(f'*{search_term}*'):
                stat = file_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                modified = datetime.fromtimestamp(stat.st_mtime)
                print(f"Found [{dir_label}]: {file_path.name:<40} | "
                      f"size: {size_mb:6.2f} MB | "
                      f"modified: {modified.strftime('%d %B %Y %H:%M')}")
                found_files.append(file_path)

            if not found_files:
                print(f"  No PassiveFlexion files found in {dir_label} directory.\n")
                continue

            dest_folder = found_files[0].parent / folder_name
            dest_folder.mkdir(exist_ok=True)
            print(f"\n  Destination folder: {dest_folder}")

            moved, skipped = 0, 0
            for file_path in found_files:
                dest_path = dest_folder / file_path.name
                if dest_path.exists():
                    print(f"  Skipped (already exists): {file_path.name}")
                    skipped += 1
                else:
                    shutil.move(file_path, dest_path)  
                    print(f"  Moved: {file_path.name}") 
                    moved += 1

            print(f"\n  [{dir_label}] Done — {moved} file(s) moved, {skipped} skipped.\n")
            
    def rename_testbins(self, tests, compforces, pclconds, flex_angles, recursive=True):
        """
        """
        print(f"\n=== Renaming Test files for {self.subject} ===\n")

        search_dirs = {"BIN": self.bin_dir}

        for dir_label, search_dir in search_dirs.items():
            print(f"--- [{dir_label}] directory ---\n")
            total_renamed, total_skipped = 0, 0

            for test, compforce, pclcond, flex_angle in itertools.product(tests, compforces, pclconds, flex_angles):
                old_term = f"{self.subject}_{int(compforce)}_{test}_{pclcond}_{int(flex_angle)}d"
                new_term = f"{self.subject}_{test}_{int(compforce)}_{pclcond}_{int(flex_angle)}d"

                glob_fn = search_dir.rglob if recursive else search_dir.glob
                found_files = list(glob_fn(f'*{old_term}*'))

                if not found_files:
                    print(f"  No files found for: {old_term}")
                    continue

                print(f"  Found {len(found_files)} file(s) for: {old_term}")
                for file_path in found_files:
                    stat = file_path.stat()
                    print(f"    {file_path.name:<50} | "
                        f"{stat.st_size / (1024*1024):6.2f} MB | "
                        f"{datetime.fromtimestamp(stat.st_mtime).strftime('%d %B %Y %H:%M')}")

                for file_path in found_files:
                    new_name = file_path.name.replace(old_term, new_term)
                    dest_path = file_path.parent / new_name
                    if dest_path.exists():
                        print(f"  Skipped (already exists): {new_name}")
                        total_skipped += 1
                    else:
                        file_path.rename(dest_path)
                        print(f"  Renamed: {file_path.name}  -->  {new_name}\n")
                        total_renamed += 1

            print(f"\n  [{dir_label}] Done — {total_renamed} file(s) renamed, {total_skipped} skipped.\n")
            
            
    def rename_testoutput(self, tests, compforces, pclconds, flex_angles, recursive=True):
        """
        Identifies files containing '_sim_PassiveFlexion_' in their name and renames them
        by removing the '_sim' prefix, in place (no folder created).
        """
        print(f"\n=== Renaming PassiveFlexion files for {self.subject} ===\n")

        search_dirs = { "OUTPUT": self.output_dir,}

        for dir_label, search_dir in search_dirs.items():
            print(f"--- [{dir_label}] directory ---\n")
            total_renamed, total_skipped = 0, 0

            for test, compforce, pclcond, flex_angle in itertools.product(tests, compforces, pclconds, flex_angles):
                old_term = f"sim_results_{int(compforce)}_{self.subject}_{test}_{pclcond}_{int(flex_angle)}d"
                new_term = f"{self.subject}_{test}_{int(compforce)}_{pclcond}_{int(flex_angle)}d"

                glob_fn = search_dir.rglob if recursive else search_dir.glob
                found_files = list(glob_fn(f'*{old_term}*'))

                if not found_files:
                    print(f"  No files found for: {old_term}")
                    continue

                print(f"  Found {len(found_files)} file(s) for: {old_term}")
                for file_path in found_files:
                    stat = file_path.stat()
                    print(f"    {file_path.name:<50} | "
                        f"{stat.st_size / (1024*1024):6.2f} MB | "
                        f"{datetime.fromtimestamp(stat.st_mtime).strftime('%d %B %Y %H:%M')}")

                for file_path in found_files:
                    new_name = file_path.name.replace(old_term, new_term)
                    dest_path = file_path.parent / new_name
                    if dest_path.exists():
                        print(f"  Skipped (already exists): {new_name}")
                        total_skipped += 1
                    else:
                        file_path.rename(dest_path)
                        print(f"  Renamed: {file_path.name}  -->  {new_name}\n")
                        total_renamed += 1

            print(f"\n  [{dir_label}] Done — {total_renamed} file(s) renamed, {total_skipped} skipped.\n")