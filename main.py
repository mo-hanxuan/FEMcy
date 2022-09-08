import taichi as ti
from stiffnessMtrx import System_of_equations
from readInp import *
from material import *
from body import Body
from tiMath import field_abs_max


if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)
    
    fileName = input("\033[32;1m please give the .inp format's "
                        "input file path and name: \033[0m")
    ### for example, fileName = ./tests/ellip_membrane_linEle_localVeryFine.inp
    inp = Inp_info(fileName)
    nodes, eSets = inp.nodes, inp.eSets
    body = Body(nodes=nodes, elements=list(eSets.values())[0])
    ele_type = list(eSets.keys())[0]
    if ele_type[0:3] == "CPS":
        material = Linear_isotropic_planeStress(modulus=inp.materials["Elastic"][0], 
                                                poisson_ratio=inp.materials["Elastic"][1])
    elif ele_type[0:3] == "CPE":
        material = Linear_isotropic_planeStrain(modulus=inp.materials["Elastic"][0], 
                                                poisson_ratio=inp.materials["Elastic"][1])

    equationSystem = System_of_equations(body, material)

    equationSystem.solve(inp, show_newton_steps=True)
    print("\033[40;33;1m equationSystem.dof = \n{} \033[0m".format(equationSystem.dof.to_numpy()))

    ### show the body by mises stress
    equationSystem.compute_strain_stress() 
    stress = equationSystem.body.mises_stress.to_numpy()
    print("\033[35;1m maximum mises_stress = {} MPa \033[0m".format(stress.max()), end="; ")
    print("\033[40;33;1m max dof (disp) = {} \033[0m".format(field_abs_max(equationSystem.dof)))
    windowLength = 512
    gui = ti.GUI('mises stress', res=(windowLength, windowLength))
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
    stress = equationSystem.body.cauchy_stress.to_numpy()[:, :, stress_id[0], stress_id[1]]
    print("\033[35;1m maximum stress[{}] = {} MPa \033[0m".format(stress_id, abs(stress).max()), end="; ")
    print("\033[40;33;1m max dof (disp) = {} \033[0m".format(field_abs_max(equationSystem.dof)))
    gui = ti.GUI('stress[{}, {}]'.format(*stress_id), res=(windowLength, windowLength))
    while gui.running:
        equationSystem.body.show2d(gui, disp=equationSystem.dof, 
                                   field=stress)
