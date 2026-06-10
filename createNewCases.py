# -*- coding: utf-8 -*-
"""
createFolderStructure.py
Reza Pourmodheji - Jan 2026

Creates standardized folder structure for TKA modeling projects
"""

import os
from pathlib import Path
import shutil

# ── Configuration ──────────────────────────────────────────────────────────────
subjects = ['TKAS02', 'TKAS03', 'TKAS04', 'TKAS05', 'TKAS06', 'TKAS07', 'TKAS08', 'TKAS09', 'TKAS10', 'TKAS11']
# subjects = ['TKAS02']  # For testing, comment out when ready for all subjects
cases = ['CS', 'PS']


# Base study path (Windows style)
# study = Path(r'S:\BiomechanicsResearch/groupImhauser/Modeling/TKA modeling projects\Zimmer Persona PS - CPS -UC - MC/PFJ-Inverse-Kinematics')
study = Path(r'C:/temp/MCL-Sensitivity/')
for subject in subjects:
    for case in cases:
        print(f"Processing {subject} - {case}")
        case_path = study / 'Data_Reduced' / 'Subjects' / subject / case
        case_new_path = study / 'Data_Reduced' / 'Subjects' / subject / f"{case} - collected"
        shutil.copytree(case_path, case_new_path, dirs_exist_ok=True)