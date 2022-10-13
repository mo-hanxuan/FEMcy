import taichi as ti
from stiffnessMtrx import System_of_equations
from readInp import *
from material import *
from body import Body
from tiMath import field_abs_max, scalerField_from_matrixField, vectorField_max
import time


if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)
    
    fileName = input("\033[32;1m please give the .inp format's "
                        "input file path and name: \033[0m")  # e.g., find an .inp file in tests folder and paste its path here
    inpName = fileName.split("/")[-1] if "/" in fileName else fileName.split("\\")[-1]
    inpName = inpName.split(".inp")[0]
    
    inpPath = fileName[: -len(inpName+".inp")]

    ### use the inp file to apply finite element analysis
    inp = Inp_info(fileName)
    nodes, eSets = inp.nodes, inp.eSets
    body = Body(nodes=nodes, elements=list(eSets.values())[0], ELE=inp.ELE)
    material = list(inp.materials.values())[0]

    equationSystem = System_of_equations(body, material, inp.geometric_nonlinear)

    time0 = time.time()
    equationSystem.solve(inp, show_newton_steps=True, save2path=None)  # save2path=inpPath+inpName)
    time1 = time.time(); 
    print("\033[40;33;1m equationSystem.dof = \n{}, time for finite element computing (include compling) is {} s \033[0m".format(
        equationSystem.dof.to_numpy(), time1 - time0))

    ### show the body by mises stress
    equationSystem.compute_strain_stress() 
    stress = equationSystem.mises_stress.to_numpy()
    print("\033[35;1m max mises_stress at integration point is {} MPa \033[0m".format(stress.max()), end="; ")
    print("\033[40;33;1m max dof (disp) = {} \033[0m".format(field_abs_max(equationSystem.dof)))
    windowLength = 512
    if not isinstance(equationSystem.ELE, Element_linear_triangular):  # situation when using 3D-GUI
        equationSystem.ELE.extrapolate(equationSystem.mises_stress, equationSystem.nodal_vals)
        print("\033[35;1m max nodal mises_stress = {} \033[0m".format(vectorField_max(equationSystem.nodal_vals)))
        window = ti.ui.Window('Mises stress', (windowLength, windowLength))
        while window.running:
            equationSystem.body.show(window, equationSystem.dof, equationSystem.nodal_vals)
    else:  # situation when using 2D-GUI
        gui = ti.GUI('mises stress', res=(windowLength, windowLength))
        equationSystem.body.show2d(gui, disp=equationSystem.dof, 
                                    field=stress, 
                                    save2path=inpPath+"MisesStress_{}.png".format(inpName))
        while gui.running:
            equationSystem.body.show2d(gui, disp=equationSystem.dof, 
                                    field=stress)
    
    ### show other stresses
    stress_id = int(input("\033[32;1m {} \033[0m".format(
        "which stress do you want to show: \n"
        "0: σxx, 1: σyy, 2: σxy\n"
        "stress index = "
    )))
    stress_id = {0: (0, 0), 1: (1, 1), 2: (0, 1)}[stress_id]
    stress = equationSystem.cauchy_stress.to_numpy()[:, :, stress_id[0], stress_id[1]]
    print("\033[35;1m maximum stress[{}] = {} MPa \033[0m".format(stress_id, abs(stress).max()), end="; ")
    print("\033[40;33;1m max dof (disp) = {} \033[0m".format(field_abs_max(equationSystem.dof)))
    
    if not isinstance(equationSystem.ELE, Element_linear_triangular):  # situation when using 3D-GUI
        window.destroy()
        window = ti.ui.Window('stress[{}, {}]'.format(*stress_id), (windowLength, windowLength))
        scalerField_from_matrixField(equationSystem.visualize_field, equationSystem.cauchy_stress, *stress_id)
        equationSystem.ELE.extrapolate(equationSystem.visualize_field, equationSystem.nodal_vals)
        print("\033[35;1m max nodal stress[{}, {}] = {} \033[0m".format(*stress_id, vectorField_max(equationSystem.nodal_vals)))
        while window.running:
            equationSystem.body.show(window, equationSystem.dof, equationSystem.nodal_vals)
    else: 
        gui = ti.GUI('stress[{}, {}]'.format(*stress_id), res=(windowLength, windowLength))
        equationSystem.body.show2d(gui, disp=equationSystem.dof, 
            field=stress, save2path=inpPath+"stress{}_{}.png".format(stress_id, inpName))
        while gui.running:
            equationSystem.body.show2d(gui, disp=equationSystem.dof, 
                                    field=stress)
