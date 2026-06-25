# -*- coding: utf-8 -*-
"""
createFolderStructure.py
Reza Pourmodheji - June 2026

Creates standardized folder structure for TKA modeling projects
"""

import os
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

# Base study path (Windows style)
study = Path(r'S:\BiomechanicsResearch\groupImhauser\OREF TKA\Modeling')
subjects = ['S044']

# ── Main ───────────────────────────────────────────────────────────────────────
def create_subject_structure():
    created_count = 0
    
    for subject in subjects:
        # Main subject directory
        base_path = os.path.join(
            study,
            'Data_Reduced',
            subject,
        )
        
        # Create main folder
        os.makedirs(base_path, exist_ok=True)
        created_count += 1 if not os.path.exists(base_path) else 0
        
        print(f"Created/checked: {base_path}")
        
        # You can uncomment and extend the subfolder structure as needed
        subfolders = [
            'bin_models',
            'contact_incidents',
            'Macros and CMD',
            'model_inputs',
            'model_outputs'
        ]
        
        # Optional deeper structure
        deeper = {
            'model_inputs': [
                'Geometries',
                'Transformations',
                'Lig_update',
                'ligOpt' ,
                'Motions' ,
                'MakoPoints',
                'Screenshots',
                'CT_data'
            ],
            'model_outputs': [
                'Ligaments'
            ]
        }
        
        for folder in subfolders:
            path = os.path.join(base_path, folder)
            os.makedirs(path, exist_ok=True)
            
            # Create deeper folders when applicable
            if folder in deeper:
                for sub in deeper[folder]:
                    os.makedirs(os.path.join(path, sub), exist_ok=True)

if __name__ == "__main__":
    print("Starting TKA subject folder structure creation...")
    print(f"Base study path: {study}\n")
    
    create_subject_structure()
    
    print("All done ✓")