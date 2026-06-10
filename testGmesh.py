import gmsh
import sys
import math

gmsh.initialize()
gmsh.clear()

# 1. Load the STL
gmsh.merge("S:\BiomechanicsResearch\groupImhauser\OREF TKA\Modeling\Data_Reduced\S026\model_inputs\CT_data\Pt20.stl")

# 2. Reclassify surfaces (detect sharp edges)
angle = 90  # Angle for surface detection in degrees
gmsh.model.mesh.classifySurfaces(angle * math.pi / 180., True, True, math.pi)

# 3. Create topology to enable meshing
gmsh.model.mesh.createTopology()

# 4. Generate the new mesh (2D) and write output
gmsh.model.mesh.generate(2)
gmsh.write("S:\BiomechanicsResearch\groupImhauser\OREF TKA\Modeling\Data_Reduced\S026\model_inputs\CT_data\Pt20_remesh.stl")

gmsh.finalize()
