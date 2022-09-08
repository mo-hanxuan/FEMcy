# FEMcy    
## an open-source **finite element** solver with cross-platform **parallel** (CPU/**GPU**) computing
---
FEMcy is a finite element solver for **structural static/dynamic analysis** in **continuum mechanics**, powered by cross-platform parallel (CPU/GPU) computing language of **Taichi**. FEMcy provides an alternative option besides Abaqus. Compared to the widely-used finite element softwares such as Abaqus, Ansys and COMSOL, we present the FEMcy which is flexible for customized needs by open-source. The conventional black-box of computational structural analysis (**CSD**) is now opened for you to stare at the mechanism behind it, and manipulate it to fit your customized needs. Compared to the open-source finite element softwares such as CalculiX or OOFEM, we provide the implementation on GPU parallel computing, meanwhile maintain the friendly readability by Python language. 
---
## Features
+ both small deformation and large deformation (geometric nonlinearity) are enabled
+ friendly readability by Python, parallel by **Taichi**
+ material nonlinearity (customize your constitutive model)
+ many types of elements including second-order elements
+ Dirichlet boundary condition (BC) and Neumann BC are enabled

## Installation and Usage
1. install Python (3.8+) and pip, then install Taichi and numpy
> pip install numpy <br>
> pip install taichi
2. git clone this project
3. go to the current directory of this project, and run the code by:
> python ./main.py
4. Pre-processing: choose an .inp file (the Abaqus-input-file-format) which defines the geometry, mesh, material and boundary condition for structual analysis. The .inp file can be obtained by Abaqus pre-processing GUI. Insert the path and inp file name to the command line:
> 
## Examples (some benchmark problems) 
### 1. ellip-membrane with constant pressure

### 2. comparison of small and large deformation by beam-deflection

### 3. Cook's membrane

## Future work
+ multiphysics, general PDE solver
+ dynamic analysis
+ contact and friction (maybe powered by the fasionable IPC (incremental potential contact) someday)
+ flexible adaptive-mesh (local-refinement dynamically)
+ more types of boundary conditions, such as periodic-boundary-condition (PBC) by Lagrangian-multiplier method is on-going
+ support more file-formats for pre-processing, such as Ansis input file format. Or even develop a pre-processing GUI for you to define the geometry, mesh, material, boundary conditions, etc.
+ support more sophisticated post-processing of the output data, such as output data file for VTK visulization or ParaView visulization. 

## References

