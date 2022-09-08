
import taichi as ti
from stiffnessMtrx import System_of_equations
from readInp import *
from material import *
from body import Body
from tiMath import field_abs_max


if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)
    
    fileName = input("\033[32;1mm please give the .inp format's "
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
    # equationSystem.get_dnatdxs()
    # print("\033[35;1m equationSystem.dnatdxs = {}\033[0m".format(equationSystem.dnatdxs))

    equationSystem.solve(inp, geometric_nonlinear=True, show_newton_steps=True)
    print("\033[40;33;1m equationSystem.dof = \n{} \033[0m".format(equationSystem.dof.to_numpy()))

    for i in range(equationSystem.rhs.shape[0]):
        print("\033[32;1m rhs, nodal force = {}, {} \033[0m".format(
            equationSystem.rhs[i], equationSystem.nodal_force[i]
        ), end="; ")

    ### show the body
    equationSystem.compute_strain_stress()
    stress_id = int(input("\033[32;1m {} \033[0m".format(
        "which stress do you want to show: \n"
        "0: σxx, 1: σyy, 2: σxy\n"
        "stress index = "
    )))
    stress = equationSystem.body.stresses.to_numpy()[:, :, stress_id]
    print("\033[35;1m maximum stress[{}] = {} MPa \033[0m".format(stress_id, abs(stress).max()), end="; ")
    print("\033[35;1m maximum cauchy_stress[{}] = {} MPa \033[0m".format(
        stress_id, abs(equationSystem.body.cauchy_stress.to_numpy()[:, :, stress_id]).max()))
    print("\033[40;33;1m max dof (disp) = {} \033[0m".format(field_abs_max(equationSystem.dof)))
    
    windowLength = 512
    gui = ti.GUI('show body', res=(windowLength, windowLength))
    while gui.running:
        equationSystem.body.show2d(gui, disp=equationSystem.dof, field=stress)