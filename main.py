import taichi as ti
from stiffnessMtrx import System_of_equations
from readInp import *
from material import *
from body import Body
from tiGadgets import field_abs_max, scalerField_from_matrixField, vectorField_max
import time, os; os.system("")


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f64)
    
    ### input abaqus .inp format file, including file path and name
    fileName = input("\033[32;1m please give the .inp format's "
                        "input file path and name: \033[0m")  
    inpName = fileName.split("/")[-1] if "/" in fileName else fileName.split("\\")[-1]
    inpName = inpName.split(".inp")[0]
    inpPath = fileName[: -len(inpName+".inp")]

    ### use the inp file to apply finite element analysis
    inp = Inp_info(fileName)
    nodes, eSets = inp.nodes, inp.eSets
    body = Body(nodes=nodes, elements=list(eSets.values())[0], ELE=inp.ELE)
    material = list(inp.materials.values())[0]

    system = System_of_equations(body, material, inp.geometric_nonlinear)

    time0 = time.time()
    system.solve(inp, show_newton_steps=True, save2path=None) # save2path=inpPath+inpName)
    time1 = time.time(); 
    print(f"\033[40;33;1m system.dof = \n{system.dof.to_numpy()}, "
          f"time for finite element computing (include compling) is {time1 - time0} s \033[0m")
    
    system.get_elasEng()
    print(f"total elastic energy is {system.elsEng}")

    ### show the body by mises stress
    system.compute_strain_stress() 
    stress = system.mises_stress.to_numpy()
    print(f"\033[35;1m max mises_stress at integration point is {stress.max()} MPa \033[0m", end="; ")
    print(f"\033[40;33;1m max dof (disp) = {field_abs_max(system.dof)} \033[0m")
    windowLength = 512
    system.ELE.extrapolate(system.mises_stress, system.nodal_vals)
    print(f"\033[35;1m max nodal mises_stress = {vectorField_max(system.nodal_vals)} \033[0m")
    window = ti.ui.Window('Mises stress', (windowLength, windowLength))
    while window.running:
        system.body.show(window, system.dof, system.nodal_vals)
    
    ### show other stresses
    if system.dm == 2:  # case of 2D
        stress_id = int(input("\033[32;1m {} \033[0m".format(
            "which stress do you want to show: \n"
            "0: σxx, 1: σyy, 2: σxy\n"
            "stress index = "
        )))
        assert stress_id in [0, 1, 2]
        stress_id = {0: (0, 0), 1: (1, 1), 2: (0, 1)}[stress_id]
    else:  # case of 3D
        stress_id = int(input("\033[32;1m {} \033[0m".format(
            "which stress do you want to show: \n"
            "0: σxx, 1: σyy, 2: σzz, 3: σxy, 4: σzx, 5: σyz\n"
            "stress index = "
        )))
        assert stress_id in [0, 1, 2, 3, 4, 5]
        stress_id = {0: (0, 0), 1: (1, 1), 2: (2, 2),  # Voigt notation
                     3: (0, 1), 4: (2, 0), 5: (1, 2)}[stress_id]
    stress = system.cauchy_stress.to_numpy()[:, :, stress_id[0], stress_id[1]]
    print(f"\033[35;1m maximum stress[{stress_id}] = {abs(stress).max()} MPa \033[0m", end="; ")
    print(f"\033[40;33;1m max dof (disp) = {field_abs_max(system.dof)} \033[0m")
    
    window.destroy()
    window = ti.ui.Window(f'stress{stress_id}', (windowLength, windowLength))
    scalerField_from_matrixField(system.visualize_field, system.cauchy_stress, *stress_id)
    system.ELE.extrapolate(system.visualize_field, system.nodal_vals)
    print(f"\033[35;1m max nodal stress{stress_id} = {vectorField_max(system.nodal_vals)} \033[0m")
    while window.running:
        system.body.show(window, system.dof, system.nodal_vals)
