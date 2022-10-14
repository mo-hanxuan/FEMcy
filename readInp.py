"""
    read the .inp file (Abaqus input file format) and 
    get the info about the geometric model, mesh, boundary condition, material, etc.
"""
import numpy as np
import taichi as ti
import os; import copy; import sys
from element_linear_quadrilateral import Element_linear_quadrilateral
from element_linear_tetrahedral import Element_linear_tetrahedral
from element_linear_triangular import Element_linear_triangular
from element_quadratic_quadrilateral import Element_quadratic_quadrilateral
from element_quadratic_tetrahedral import Element_quadratic_tetrahedral
from element_quadratic_triangular import Element_quadratic_triangular
from material import *


class Inp_info(object):
    """extract information from .inp file 
    (using Abaqus or CalculiX's input file format)"""
    
    def __init__(self, file) -> None:
        self.nodes, self.eSets = self.read_node_element(file)
        self.node_sets, self.ele_sets = self.read_set(file)  # sub-set
        self.face_sets = self.read_face_set(file)
        self.dirichlet_bc_info, self.neumann_bc_info = self.get_boundary_condition(file)
        self.materials = self.read_material(file)
        self.geometric_nonlinear = self.read_geometric_nonlinear(file)
        self.time_incs = self.read_time_inc(file)


    def read_node_element(self, fileName='donut.inp'):
        """
        read the inp file, returns:
            nodes: the coordinates of all nodes
            elements: corresponding node numbers of all elements
        """

        nodes = {}
        cout = False
        with open(fileName, 'r') as file:
            for line in file:
                if '*' in line:
                    if cout: break
                
                if cout:
                    data = line.split(',')
                    data = list(map(float, data))
                    nodes[int(data[0])] = data[1:]
                
                if '*Node' in line or '*NODE' in line or '*node' in line:
                    cout = True

        elements = np.array([], dtype=np.int64)
        cout = False
        text = {}
        with open(fileName, 'r') as file:
            for line in file:
                if '*' in line:
                    cout = False
                
                if cout:
                    data = line[:-1].rstrip().rstrip(',')
                    data = data.split(',')
                    tex = []
                    for x in data:
                        tex.append(x)
                    text[currentType].extend(tex)
                
                if '*ELEMENT' in line or '*Element' in line or '*element' in line:
                    for type_ in ["C3D8", "C3D20", "C3D4", "C3D10", "B31", "C3D6", 
                                "CPS3", "CPE3", "CPE4", "CPS4", "CPE8", "CPS8", 
                                "CPS6", "CPE6"]:  # the surported element types
                        if ("TYPE=" in line or "type=" in line) and type_ in line:
                            if type_ not in text:
                                text[type_] = []
                            currentType = type_
                            cout = True
                            break
        if len(text) > 1:
            print("\033[31;1m there are multiple element types in the file, \033[0m")
            print("\033[40;33;1m {} \033[0m".format(list(text.keys())))
        eSets = {}
        for eType in text:
            data = list(map(int, text[eType]))
            elements = np.array(data)
            if eType == "C3D8":
                elements = elements.reshape((-1, 9))
                elements = elements[:, 1:]
            elif eType == "C3D20":
                elements = elements.reshape((-1, 21))
                elements = elements[:, 1:9]
            elif eType in ["C3D4", "CPE4", "CPS4"]:
                elements = elements.reshape((-1, 5))
                elements = elements[:, 1:]
            elif eType in ["CPS8", "CPE8"]:
                elements = elements.reshape((-1, 9))
                elements = elements[:, 1:]
            elif eType == "C3D10":
                elements = elements.reshape((-1, 11))
                elements = elements[:, 1:11]
            elif eType == "B31":
                elements = elements.reshape((-1, 3))
                elements = elements[:, 1:]
            elif eType == "CPS3" or eType == "CPE3":
                    elements = elements.reshape((-1, 4))
                    elements = elements[:, 1:]
            elif eType == "C3D6":
                    elements = elements.reshape((-1, 7))
                    elements = elements[:, 1:]
            elif eType == "CPS6" or eType == "CPE6":
                    elements = elements.reshape((-1, 7))
                    elements = elements[:, 1:]
            else:
                print("\033[31;1m Error, element type {} is not found! \033[0m".format(eType))
                sys.exit(1)
            eSets[eType] = elements
        
        ### transform the dictionary to np.ndarray, so that all index starts with 0
        nodes, eSets = self.sequence_order_of_body(nodes, eSets)

        ele_types = {"CPE3": Element_linear_triangular, "CPS3": Element_linear_triangular,
                     "CPE4": Element_linear_quadrilateral, "CPS4": Element_linear_quadrilateral, 
                     "CPS6": Element_quadratic_triangular, "CPE6": Element_quadratic_triangular,
                     "CPS8": Element_quadratic_quadrilateral, "CPE8": Element_quadratic_quadrilateral,
                     "C3D4": Element_linear_tetrahedral, "C3D10": Element_quadratic_tetrahedral}
        self.ELE = ele_types[list(eSets.keys())[0]]()

        if len(eSets) == 1:
            return nodes, eSets
        else:
            raise ValueError("\033[31;1m multiple element types have not been supported now \033[0m")


    def read_set(self, fileName='donut.inp'):
        """read the inp file, get the node set, element set"""
        node_sets, ele_sets = {}, {}
        with open(fileName, 'r') as file:
            reading_data = False
            for line in file:
                if line[0:2] == "**":  # skip the notes line
                    continue
                if line[0] == "*":
                    splited_line = line.split(",")
                    set_type = splited_line[0]
                    if set_type in ["*Nset", "*Elset"] and "instance" in line:
                        ### now we get the corresponding set
                        if set_type == "*Nset":
                            sets = node_sets
                        elif set_type == "*Elset":
                            sets = ele_sets
                        set_name = splited_line[1].split("=")[1]
                        sets[set_name] = set()
                        generate = True if "generate" in splited_line[-1] else False
                        reading_data = True; continue
                    else:
                        reading_data = False
                if reading_data:
                    ### insert the data into set
                    try:
                        data = list(map(int, line.split(",")))
                    except:
                        data = list(map(int, line.split(",")[:-1]))
                    if generate:
                        sets[set_name] |= {*np.arange(data[0], data[1] + data[2], data[2])}
                    else:
                        sets[set_name] |= {*data}
        ### the number should start from 0 
        for set_ in node_sets:
            node_sets[set_] = np.array([*node_sets[set_]]) - 1
        for set_ in ele_sets:
            ele_sets[set_] = np.array([*ele_sets[set_]]) - 1
        return node_sets, ele_sets


    def read_face_set(self, fileName='./donut.inp'):
        """read the inp file, get the node set, element set"""
        if not hasattr(self, "eSets"):
            self.nodes, self.eSets = self.read_node_element(file)
        eSets = self.eSets
        face_sets = {}
        with open(fileName, 'r') as file:
            reading_data = False
            for line in file:
                if line[0:2] == "**":  # skip the notes line
                    continue
                if line[0] == "*":
                    line = line.split("\n")[0]
                    splited_line = line.split(",")
                    if splited_line[0] == "*Surface":
                        ### now we get the corresponding set
                        set_name = splited_line[2].split("=")[1]
                        face_sets[set_name] = []
                        reading_data = True; continue
                    else:
                        reading_data = False
                if reading_data:
                    ### insert the data into set
                    line = line.split("\n")[0]
                    splited_line = line.split(",")
                    face_sets[set_name].append({"ele_set": splited_line[0], "face_num": splited_line[1]})
        
        ### unfold the face set
        node_sets, ele_sets = self.read_set(fileName)
        ele_type = list(eSets.keys())[0]
        ele = self.ELE
        face2node = ele.inp_surface_num  # face to nodes
        for face_set in face_sets:
            dictList = copy.deepcopy(face_sets[face_set])
            face_sets[face_set] = set()
            for fset in dictList:
                fnum = int(fset["face_num"].split("S")[1]) - 1  # fnum starts from 0
                for iele in ele_sets[fset["ele_set"]]:
                    for local_nodes in face2node[fnum]:
                        global_nodes = (eSets[ele_type][iele][local_node] for local_node in local_nodes)
                        face_sets[face_set].add(tuple(sorted(global_nodes)))  # modified later             
        return face_sets


    def get_boundary_condition(self, fileName):
        if not hasattr(self, "node_sets"):
            self.node_sets, self.ele_sets = self.read_set(fileName) 
        if not hasattr(self, "face_sets"):
            self.face_sets = self.read_face_set(fileName)

        ### get the dirichlet boundary condition
        dirichlet_bc_info = []
        with open(fileName, "r") as file:
            reading_data = False
            for line in file:
                if line[0:2] == "**":  # skip the notes line
                    continue
                if line[0] == "*":
                    if line[0:9] == "*Boundary":
                        reading_data = True
                        if "user" in line: user = True
                        else: user = False
                        continue
                    else:
                        reading_data = False
                if reading_data:
                    ### now we get the corresponding BC info
                    splited_line = line.split("\n")[0].split(",")
                    set_name = splited_line[0]
                    dof = int(splited_line[1])  # degree of freedom
                    disp = float(splited_line[3]) if len(splited_line) >= 4 else 0.
                    dirichlet_bc_info.append(
                        {"node_set": self.node_sets[set_name], "dof": dof - 1, "val": disp, "user": user})
        
        ### get the Neumann boundary condition
        neumann_bc_info = []
        with open(fileName, "r") as file:
            reading_data = False
            for line in file:
                if line[0:2] == "**":  # skip the notes line
                    continue
                if line[0] == "*":
                    if line[0:7] == "*Dsload":
                        reading_data = True; continue
                    else:
                        reading_data = False
                if reading_data:
                    ### now we get the corresponding BC info
                    splited_line = line.split("\n")[0].split(",")
                    set_name = splited_line[0]
                    if len(splited_line) <= 3:  # pressure load
                        surface_traction = -float(splited_line[2])  # traction is negative direction of pressure
                        neumann_bc_info.append(
                            {"face_set": self.face_sets[set_name], "traction": surface_traction})
                    else:  # instruct the direction of traction force
                        surface_traction = float(splited_line[2])
                        direction = list(map(float, splited_line[3:6]))
                        neumann_bc_info.append(
                            {"face_set": self.face_sets[set_name], "traction": surface_traction, 
                            "direction": np.array(direction)})
        return dirichlet_bc_info, neumann_bc_info
    

    def read_material(self, fileName='./donut.inp'):
        materials = {}
        with open(fileName, "r") as file:
            previous_line = None
            for line in file:
                if line[0:2] == "**":  # skip the notes line
                    continue
                if line[0] == "*":
                    if line[0:9] == "*Material":
                        previous_line = "*Material"; continue
                if previous_line == "*Material":
                    material_type = line.split("*")[1].split("\n")[0]
                    previous_line = "material_type"; continue
                if previous_line == "material_type":
                    if line[0] != "*":
                        splited_line = line.split("\n")[0].split(",")
                        materials[material_type] = list(map(float, splited_line))
                    else:
                        previous_line = None; continue
        ### deduce material by element and material type
        ele_type = list(self.eSets.keys())[0]
        if ele_type[0:3] in ["CPS", "CPE"]:  # 2D case
            for key in materials:
                if key != "Elastic":
                    raise ValueError("only support linear elastic material for 2d element now.")
                else:
                    if ele_type[0:3] == "CPS":
                        materials[key] = Linear_isotropic_planeStress(modulus=materials["Elastic"][0], 
                                                                poisson_ratio=materials["Elastic"][1])
                    elif ele_type[0:3] == "CPE":
                        materials[key] = Linear_isotropic_planeStrain(modulus=materials["Elastic"][0], 
                                                                poisson_ratio=materials["Elastic"][1])
        elif ele_type[0:3] == "C3D":
            for key in materials:
                if key == "Elastic":
                    materials[key] = Linear_isotropic(modulus=materials["Elastic"][0], 
                                                poisson_ratio=materials["Elastic"][1])
                elif "neo hooke" in key:
                    materials[key] = NeoHookean(C1=materials[key][0], D1=1./materials[key][1])
                else:
                    raise ValueError("material type {} has not been supported now".format(key))
        return materials
    

    def read_geometric_nonlinear(self, fileName: str='./donut.inp') -> bool:
        """read the inp file and get whether geometric nonlinear is on"""
        with open(fileName, "r") as file:
            for line in file:
                if line[:5] == "*Step":
                    line = line.split("\n")[0].split(",")[-1].split("nlgeom=")[-1]
                    if line == "NO":
                        geometric_nonlinear = False
                    else:
                        geometric_nonlinear = True
                    break
        return geometric_nonlinear
    

    def read_time_inc(self, fileName: str):
        """read time increment"""
        with open(fileName, "r") as file:
            reading_data = False
            for line in file:
                if line[:7] == "*Static":
                    reading_data = True; continue
                if reading_data:
                    if line[0:2] == "**":  # skip the notes line
                        continue
                    line = line.split("\n")[0].split(",")
                    line = list(map(float, line))
                    time_incs = {"ini_inc": line[0], "max_time": line[1], 
                                 "min_inc": line[2], "max_inc": line[3]}
                    break
        if time_incs["ini_inc"] > time_incs["max_inc"]:
            time_incs["ini_inc"] = time_incs["max_inc"]
        return time_incs    
            

    def sequence_order_of_body(self, nodes, eSets):
        """
            change the nodes from dictionary to a list
            element nodes should be changed correspondingly
        """
        key2id = {}; nodeArray = []
        id = 0
        for key in nodes:
            key2id[key] = id
            nodeArray.append(nodes[key])
            id += 1
        for eType in eSets:
            eSet = eSets[eType]
            for ele in range(len(eSet)):
                eSet[ele, :] = np.array([key2id[key] for key in eSet[ele]])
        return np.array(nodeArray), eSets


if __name__ == "__main__":
    os.system("")
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)
    file = input("\033[32;1m please give the file path/name: \033[0m")
    inp = Inp_info(file)
    print("nodes = \n{}\n  eSets = \n{}".format(inp.nodes, inp.eSets))

    for eType in inp.eSets:
        print("\033[32;1m node number, eSets[{}].max() = {}, eSets[{}].min() = {} \033[0m".format(
            eType, inp.eSets[eType].max(), eType, inp.eSets[eType].min()
        ))

    print("\033[32;1m node_sets = {} \033[0m".format(inp.node_sets))
    print("\033[31;1m ele_sets = {} \033[0m".format(inp.ele_sets))
    print("\033[35;1m face_sets = {} \033[0m".format(inp.face_sets))

    print("\033[40;33;1m dirichlet_bc_info = {} \033[0m".format(inp.dirichlet_bc_info))
    print("\033[40;32;1m neumann_bc_info = {} \033[0m".format(inp.neumann_bc_info))
    print("\033[40;32;1m materials = {}\033[0m".format(inp.materials))
