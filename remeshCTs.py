import geomagic.app.v3
from geomagic.app.v3.imports import *

import os
from os import path
import xlsxwriter
import numpy as np
import pandas
import shutil
from pathlib import Path

'''This is a script that remeshes the CT data and saves the remeshed data in the Data_Reduced folder'''
'''Created by Reza Pourmodheji 03/08/2025'''

print(' ----------------------------------------------------------')
print(' ------------------ Running the Code ----------------------')
print(' ----------------------------------------------------------')
print('															  ')
print('															  ')


# ── PATHS ──────────────────────────────────────────────────────────────────
study     = 'S:\\BiomechanicsResearch\\groupImhauser\\OREF TKA\\Modeling'
data_raw  = study + '\\Data_Raw' 
data_reduced = study + '\\Data_Reduced'

subjects = ['S026']
bone_parts = ['Fem', 'Tib', 'Fib']
implant_parts = ['FemComp', 'Tray', 'Insert']
partsmeshsize = {'Fem': 2, 'Tib': 2, 'Fib': 1}
# partstriangles = {'FemComp': 25000, 'Fem': 12000, 'Tib': 12000, 'Fib': 8000, 'Tray': 20000, 'Insert': 25000}
partstriangles = {'Fem': 12000, 'Tib': 12000, 'Fib': 8000}
mm2m = 0.001

geo.new()

for subject in subjects:
    ct_data = data_reduced + '\\' + subject + '\\model_inputs\\CT_data'
    for part in bone_parts:
        print(ct_data + '\\' + f'{subject}_{part}.stl')
        geo.open(0, 1,  ct_data + '\\' + f'{subject}_{part}.stl')
        activeModel = geoapp.getActiveModel()
        mesh = geoapp.getMesh(activeModel)
        print(mesh)
        print(mesh.numTriangles)
        #target_triangles = round(float(mesh.numTriangles)/2)
        target_triangles = partstriangles[part]
        print(part + ' has ' + str(target_triangles) + ' target triangles.')
        reduct_factor = float(target_triangles)/float(mesh.numTriangles) * 100
        print('reduct_factor', reduct_factor)
        # geo.mesh_doctor("smallcompsize", 0.0069355, "smalltunnelsize", 0.0034677, "holesize", 0.0034677,\
        #                 "spikesens", 50, "spikelevel", 0.5, "defeatureoption", 2, "fillholeoption", 2, \
        #                 "autoexpand", 2, "operations","IntersectionCheck+", "SmallComponentCheck+",\
        #                     "SpikeCheck+", "HighCreaseCheck+", "Update", "Auto-Repair")
        geo.remesh(0.002, 0, 1, 45, partsmeshsize[part]*mm2m, 1, 0, 0, 0.002, 1) #was 0.001
        #geo.quick_smooth()
        geo.decimate_polygons(1, reduct_factor,  reduct_factor, 0.0002, 10000, 0, 0, 3, 1, 3, 2, -1, -1, 3, target_triangles)

        # geo.remesh(0.001, 0, 1, 45, partsmeshsize[part]*mm2m, 1, 0, 0, partsmeshsize[part]*mm2m, 1)
        geo.saveas(( data_reduced + '\\' + f'{subject}\\model_inputs\\Geometries' + '\\' + f'{part}.stl' ), 3, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, -1, 0, 1, 0)
        print('Saved resected ' + part + ': ' + data_reduced + f'\\{subject}\\model_inputs\\Geometries' + '\\' + f'{subject}_{part}.stl' + ' with ' + str(target_triangles) + ' triangles.')
    for part in implant_parts:
        print(ct_data + '\\' + f'{part}.stl')
        geo.open(0, 1,  ct_data + '\\' + f'{subject}_{part}.stl')
        activeModel = geoapp.getActiveModel()
        mesh = geoapp.getMesh(activeModel)
        geo.saveas(( data_reduced + '\\' + f'{subject}\\model_inputs\\Geometries' + '\\' + f'{part}.stl' ), 3, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, -1, 0, 1, 0)
        print('Saved resected ' + part + ': ' + data_reduced + f'\\{subject}\\model_inputs\\Geometries' + '\\' + f'{part}.stl')
geo.new()