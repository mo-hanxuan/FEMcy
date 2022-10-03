"""
    Body is a class constructed by nodes and elements
"""
import numpy as np
import taichi as ti

import tiMath
from colorBar import getColor

from readInp import Inp_info
from element_linear_triangular import Element_linear_triangular
from element_linear_tetrahedral import Element_linear_tetrahedral
from element_quadratic_triangular import Element_quadratic_triangular
from element_quadratic_tetrahedral import Element_quadratic_tetrahedral


@ti.data_oriented
class Body:
    def __init__(self, nodes: np.ndarray, elements: np.ndarray) -> None:
        self.nodes = ti.Vector.field(len(nodes[0]), dtype=ti.f64, shape=(len(nodes), ))  # coordinates of all nodes
        self.elements = ti.Vector.field(len(elements[0]), dtype=ti.i32, shape=(len(elements), ))  # node number of each element
        self.nodes.from_numpy(nodes)
        self.elements.from_numpy(elements)
        self.np_nodes = nodes; self.np_elements = elements
        self.dm = self.nodes[0].n
        self.disp = ti.Vector.field(len(self.nodes[0]), ti.f64, shape=(len(nodes), ), needs_grad=True)

        ### must be modified latter (modified by using element type as key to get ELE)
        if self.np_elements.shape[1] == 3:
            self.ELE = Element_linear_triangular(); print("\033[32;1m Element_linear_triangular is used \033[0m")
        elif self.np_elements.shape[1] == 6:
            self.ELE = Element_quadratic_triangular(); print("\033[32;1m Element_quadratic_triangular is used \033[0m")
        elif self.np_elements.shape[1] == 4:
            self.ELE = Element_linear_tetrahedral(); print("\033[32;1m Element_quadratic_triangular is used \033[0m")
        elif self.np_elements.shape[1] == 10:
            self.ELE = Element_quadratic_tetrahedral(); print("\033[32;1m Element_quadratic_triangular is used \033[0m")

        ### the variables for visualization
        mesh, face2ele, surfaces = self.ELE.get_mesh(self.np_elements)
        self.mesh_id = ti.field(ti.i32, shape=(surfaces.shape[0] * surfaces.shape[1])); self.mesh_id.from_numpy(surfaces.reshape(-1))
        self.mesh = ti.Vector.field(3, ti.f32, shape=(surfaces.shape[0] * surfaces.shape[1]))  # store vertex coordinates of the mesh, similar to .stl format
        mesh2ele = np.zeros(surfaces.shape[0] * surfaces.shape[1], dtype=np.int64)
        for i in range(len(surfaces)):
            for j in range(len(surfaces[i])):
                mesh2ele[i * len(surfaces[0]) + j] = list(face2ele[tuple(surfaces[i])])[0]
        self.mesh2ele = ti.field(ti.i32, shape=(surfaces.shape[0] * surfaces.shape[1],)); self.mesh2ele.from_numpy(mesh2ele)
        self.vertex_val = ti.field(ti.f64, shape=(surfaces.shape[0] * surfaces.shape[1]))
        self.vertex_color = ti.Vector.field(3, ti.f32, shape=(surfaces.shape[0] * surfaces.shape[1]))
    

    def get_surfaceEdges(self, redo=False):
        if not hasattr(self, "surfaceEdges") or redo:
            edges = set()
            for ele in self.np_elements:
                for facet in self.ELE.facet_natural_coos.keys():
                    edges.add(tuple(sorted([ele[facet[0]], ele[facet[1]]])))
            self.surfaceEdges = np.array(list(edges))

        return self.surfaceEdges


    def show2d(self, gui, disp=[], field=[], save2path: str=None):
        self.get_surfaceEdges()

        if type(disp) != type([]):
            nodes = self.np_nodes + disp.to_numpy().reshape(self.np_nodes.shape)
        else:
            nodes = self.np_nodes

        if not hasattr(self, "stretchRatio"):
            xmin = min(nodes[i][0] for i in range(self.nodes.shape[0]))
            xmax = max(nodes[i][0] for i in range(self.nodes.shape[0]))
            ymin = min(nodes[i][1] for i in range(self.nodes.shape[0]))
            ymax = max(nodes[i][1] for i in range(self.nodes.shape[0]))
            self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
            """ lengthScale is the length of the body after 
                stretch its size to match the window, 
                can not be too large otherwise your camera will be out of the scene box"""
            lengthScale = 1.
            self.stretchRatio = lengthScale / max(xmax - xmin, ymax - ymin) / 1.25  # divided by 1.25 is to save some space for deformation
        
        bottomleft = np.array([self.xmin, self.ymin])
        a, b, c, line0, line1 = self.ELE.show_triangles_2d(self.np_elements, nodes, self.surfaceEdges,
                                                           bottomleft, self.stretchRatio)
        ### get the color
        if len(field) >= 1:
            field = field.reshape(-1)
            if len(field) < len(a):  # e.g., quadratic element, color-triangles more than integration points
                field_ = np.zeros(len(a), dtype=field.dtype)
                num1, num2 = tiMath.fraction_reduction(len(field), len(a))
                field_ = field_.reshape((-1, num2)); field = field.reshape((-1, num1))
                for i in range(field.shape[0]):
                    field_[i, 0:num1] = field[i, 0:num1]
                    average_val = field[i].sum() / len(field[i])
                    field_[i, num1: num2] = average_val
                field = field_.reshape(-1)
            
            field_max, field_min = field.max(), field.min()
            field_colors = np.zeros(field.shape[0], dtype=np.int32)
            for i in range(len(field)):
                red, green, blue = getColor((field[i] - field_min) / (field_max - field_min + 1.e-30))
                field_colors[i] = ti.rgb_to_hex([red, green, blue])
        else:
            field_colors = 0xED553B

        # while gui.running:
        gui.triangles(a=a, b=b, c=c, color=field_colors)
        gui.lines(begin=line0, end=line1, radius=0.75, 
                    color=int("0x{:02x}{:02x}{:02x}".format(24, 24, 24), base=16))
        gui.show(save2path)
    

    def show(self, window: ti.ui.Window, disp, vals):

        windowLength = 1024
        lengthScale = min(windowLength, 512)  # lengthScale is the length of the body after 
                                              # stretch its size to match the window, 
                                              # can not be too large otherwise your camera will be out of the scene box
        light_distance = lengthScale / 25.
        
        if not hasattr(self, "visualizeRatio"):
            xmin = min(self.nodes[i][0] for i in range(self.nodes.shape[0]))
            xmax = max(self.nodes[i][0] for i in range(self.nodes.shape[0]))
            ymin = min(self.nodes[i][1] for i in range(self.nodes.shape[0]))
            ymax = max(self.nodes[i][1] for i in range(self.nodes.shape[0]))
            visualizeRatio = lengthScale / max(xmax - xmin, ymax - ymin) / 10.
            self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
            self.visualizeRatio = visualizeRatio
        
        center = np.array([(self.xmin + self.xmax) / 2., (self.ymin + self.ymax) / 2., 0.]) * self.visualizeRatio
        length = max(self.xmax - self.xmin, self.ymax - self.ymin)

        # window = ti.ui.Window('show body', (windowLength, windowLength))
        canvas = window.get_canvas()

        ### update the mesh and get the vertex color
        self.update_mesh(disp, self.visualizeRatio)
        self.get_vertex_val(vals)
        self.get_vertex_color()

        ### now we render the window
        if not hasattr(self, "camera"):
            camera = ti.ui.Camera(); scene = ti.ui.Scene()
            self.camera = camera; self.scene = scene
            camera.position(center[0], center[1] + 0.1 * length, 100.)  # if camera is far away from the object, you can't see any thing
            camera.lookat(center[0], center[1] + 0.1 * length, center[2])
            camera.up(0., 1., 0.)
        else:
            camera, scene = self.camera, self.scene
        camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        # if self.dm == 2: camera.projection_mode(1)
        # elif self.dm == 3: camera.projection_mode(0)
        scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
        scene.ambient_light(color=(0.5, 0.5, 0.5))
        scene.mesh(vertices=self.mesh, per_vertex_color=self.vertex_color, two_sided=True)
        canvas.scene(scene)
            
        ### show the window
        window.show()
    

    def get_nodeEles(self, redo=False):
        """get element number related to a node"""
        if not hasattr(self, "nodeEles") or redo:
            if redo:
                self.np_nodes = self.nodes.to_numpy()
                self.np_elements = self.elements.to_numpy()
            elements = self.np_elements
            nodeEles = [set() for _ in range(self.np_nodes.shape[0])]
            for iele, ele in enumerate(elements):
                for node in ele:
                    nodeEles[node].add(iele)
            for node in range(len(nodeEles)):
                nodeEles[node] = list(nodeEles[node])
            self.nodeEles = nodeEles
        return self.nodeEles

    
    def get_coElement_nodes(self, redo=False):
        """for each node, get the coElement nodes of this node"""
        if not hasattr(self, "coElement_nodes") or redo:
            self.get_nodeEles()
            coElement_nodes = []
            for node0 in range(self.np_nodes.shape[0]):
                others = set()
                for ele in self.nodeEles[node0]:
                    for node1 in self.np_elements[ele, :]:
                        others.add(node1)
                coElement_nodes.append(list(others))
            self.coElement_nodes = coElement_nodes
        return self.coElement_nodes
    

    def get_boundary(self, redo=False):  # get the boundary of this body
        if not hasattr(self, "boundary") or redo:
            facets = self.ELE.facet_natural_coos.keys() # element boundaries 
            facetDic = {}
            for iele, ele in enumerate(self.np_elements):
                for ifacet, facet in enumerate(facets):  
                    f = []
                    for node in facet:
                        f.append(int(ele[node]))
                    tmp = tuple(sorted(f))  # modified latter
                    if tmp in facetDic:
                        facetDic[tmp].append(iele)
                    else:
                        facetDic[tmp] = [iele]
            self.facetDic = facetDic
            boundary = {}
            for facet in facetDic:
                if len(facetDic[facet]) == 1:
                    boundary[facet] = facetDic[facet][0]
            self.boundary = boundary

            ### get the boundary nodes
            node2boundary = {}  # from node to boundary
            for facet in boundary:
                for node in facet:
                    if node in node2boundary:
                        node2boundary[node].add(facet)
                    else:
                        node2boundary[node] = {facet, }
            self.node2boundary = node2boundary

            ### get all nodes that belong to the boundary
            boundaryNodes = set()
            for facet in self.boundary:
                for node in facet:
                    boundaryNodes.add(node)
            self.boundaryNodes = boundaryNodes
        return self.boundary
    

    @ti.kernel 
    def init_mesh(self, ):
        elements, nodes, mesh = ti.static(self.elements, self.nodes, self.mesh)
        for ele in elements:
            for i in range(elements[ele].n):
                node = elements[ele][i]
                mesh[ele * elements[0].n + i][:nodes[node].n] = nodes[node][:nodes[node].n]


    @ti.kernel 
    def update_mesh(self, disp: ti.template(), visualizeRatio: float):
        nodes = ti.static(self.nodes)
        for i in self.mesh_id:
            node = self.mesh_id[i]
            for j in range(nodes[node].n):
                self.mesh[i][j] = nodes[node][j] + disp[node * nodes[0].n + j]
                self.mesh[i][j] *= visualizeRatio


    @ti.kernel
    def get_vertex_val(self, vals: ti.template()):
        elements, vertex_val = ti.static(self.elements, self.vertex_val)
        for vertex in vertex_val:
            ele = self.mesh2ele[vertex]
            local_i = tiMath.get_index_ti(elements[ele], self.mesh_id[vertex])
            vertex_val[vertex] = vals[ele][local_i]
    

    def get_vertex_color(self, minVal_input:float=None, maxVal_input:float=None):
        ### get the min and max val
        minVal = tiMath.field_min(self.vertex_val) if minVal_input == None else minVal_input
        maxVal = tiMath.field_max(self.vertex_val) if maxVal_input == None else maxVal_input
        ### get the color of each vertex
        self.get_vertex_color_kernel(minVal, maxVal)


    @ti.kernel
    def get_vertex_color_kernel(self, minVal: float, maxVal: float):
        for vertex in self.vertex_val:
            R, G, B = self.get_color_rainbow((self.vertex_val[vertex] - minVal) / (maxVal - minVal + 1.e-30))
            self.vertex_color[vertex] = ti.Vector([R, G, B])


    @ti.func 
    def get_color_rainbow(self, x: float):
        """input a val with 0 <= val <= 1, get the corresponding RGB color"""
        red, green, blue = 0., 0., 0.
        if x >= 0.75:
            red = 1.; green = (1. - x) / 0.25; blue = 0.
        elif 0.5 <= x < 0.75:
            red = (x - 0.5) / 0.25; green = 1.; blue = 0.
        elif 0.25 <= x < 0.5:
            red = 0.; green = 1.; blue = (0.5 - x) / 0.25
        else:
            red = 0.; green = x / 0.25; blue = 1.
        return red, green, blue


if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)
    fileName = input("\033[32;1m please give the "
                     "input file path and name: \033[0m")

    inp = Inp_info(fileName)
    nodes, eSets = inp.nodes, inp.eSets
    body = Body(nodes=nodes, elements=list(eSets.values())[0])

    # body.show()
    # body.show2d()
