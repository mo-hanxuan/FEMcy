"""
    body is a class constructed by many domains, 
    each domain contains nodes and elements,

    body has global node indexes, and global element indexes,
    domain has set of element number
"""
import numpy as np
import taichi as ti

import tiMath
from colorBar import getColor

from readInp import Inp_info
from linear_triangular_element import Linear_triangular_element
from quadritic_triangular_element import Quadritic_triangular_element


@ti.data_oriented
class Body:
    def __init__(self, nodes: np.ndarray, elements: np.ndarray) -> None:
        self.nodes = ti.Vector.field(len(nodes[0]), dtype=ti.f64, shape=(len(nodes), ))  # coordinates of all nodes
        self.elements = ti.Vector.field(len(elements[0]), dtype=ti.i32, shape=(len(elements), ))  # node number of each element
        self.nodes.from_numpy(nodes)
        self.elements.from_numpy(elements)
        self.np_nodes = nodes; self.np_elements = elements
        # self.nodes = nodes
        # self.elements = elements
        print("\033[35;1m self.nodes = {} \033[0m".format(self.nodes))
        print("\033[32;1m self.elements = {} \033[0m".format(self.elements))
        self.dm = self.nodes[0].n
        self.disp = ti.Vector.field(len(self.nodes[0]), ti.f64, shape=(len(nodes), ), needs_grad=True)

        ### must be modified latter
        if self.np_elements.shape[1] == 3:
            self.ELE = Linear_triangular_element(); print("\033[32;1m Linear_triangular_element is used \033[0m")
        elif self.np_elements.shape[1] == 6:
            self.ELE = Quadritic_triangular_element(); print("\033[32;1m Quadritic_triangular_element is used \033[0m")
    

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

        xmin = min(nodes[i][0] for i in range(self.nodes.shape[0]))
        xmax = max(nodes[i][0] for i in range(self.nodes.shape[0]))
        ymin = min(nodes[i][1] for i in range(self.nodes.shape[0]))
        ymax = max(nodes[i][1] for i in range(self.nodes.shape[0]))
        bottomleft = np.array([xmin, ymin])

        # lengthScale is the length of the body after 
        # stretch its size to match the window, 
        # can not be too large otherwise your camera will be out of the scene box
        lengthScale = 1.
        stretchRatio = lengthScale / max(xmax - xmin, ymax - ymin) 

        a, b, c, line0, line1 = self.ELE.show_triangles_2d(self.np_elements, nodes, self.surfaceEdges,
                                                           bottomleft, stretchRatio)

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
                red, green, blue = getColor((field[i] - field_min) / (field_max - field_min))
                field_colors[i] = int("0x{:02x}{:02x}{:02x}".format(
                                      int(255 * red), int(255 * green), int(255 * blue)), base=16)
        else:
            field_colors = 0xED553B

        # while gui.running:
        gui.triangles(a=a, b=b, c=c, color=field_colors)
        gui.lines(begin=line0, end=line1, radius=0.75, 
                    color=int("0x{:02x}{:02x}{:02x}".format(24, 24, 24), base=16))
        gui.show(save2path)
    

    def show(self, ):  ### currently, this is for 2d case

        xmin = min(self.nodes[i][0] for i in range(self.nodes.shape[0]))
        xmax = max(self.nodes[i][0] for i in range(self.nodes.shape[0]))
        ymin = min(self.nodes[i][1] for i in range(self.nodes.shape[0]))
        ymax = max(self.nodes[i][1] for i in range(self.nodes.shape[0]))
        center = [(xmin + xmax) / 2., (ymin + ymax) / 2., 0.]

        windowLength = 1024
        lengthScale = min(windowLength, 512)  # lengthScale is the length of the body after 
                                              # stretch its size to match the window, 
                                              # can not be too large otherwise your camera will be out of the scene box
        stretchRatio = lengthScale / max(xmax - xmin, ymax - ymin) 
        light_distance = lengthScale / 25.

        window = ti.ui.Window('show body', (windowLength, windowLength))
        canvas = window.get_canvas()

        ### if is a 2d object, then we change the node coords to 3d coords
        if self.nodes[0].n == 2:
            vertices_ = np.insert(self.nodes.to_numpy(), obj=2, values=-lengthScale, axis=1)  # insert z coordinate
            vertices = ti.Vector.field(3, ti.f32, shape=self.nodes.shape)
            vertices.from_numpy((vertices_ - center) * stretchRatio)  # put the body to the center and stretch it to fit the window
            
            indices = ti.field(ti.i32, shape=self.elements.shape[0] * 3)
            indices.from_numpy(self.elements.to_numpy().reshape(-1))

            ### now we render it
            def render():
                scene = ti.ui.Scene()
                camera = ti.ui.make_camera()
                camera.position(0., 0., lengthScale)  # if camera is far away from the object, you can't see any thing
                camera.lookat(0., 0., 0.)
                camera.up(0., 1., 0.)
                # camera.projection_mode(0)
                scene.set_camera(camera)
                scene.point_light(pos=(-light_distance, 0., light_distance), color=(0., 1., 0.))
                scene.point_light(pos=(light_distance, 0., light_distance), color=(0., 1., 0.))
                scene.ambient_light(color=(0.2, 0.2, 0.2))
                scene.mesh(vertices=vertices, indices=indices, two_sided=True)
                scene.particles(centers=vertices, radius=1.)
                canvas.scene(scene)

            ### show the window
            while True:
                render()
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
            print("\033[31;1m facets = {} \033[0m".format(facets))
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


if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)
    fileName = input("\033[32;1m please give the oofem format's "
                     "input file path and name: \033[0m")
    ### for example, fileName = D:\FEM\cases\concrete_3point\by_oofem\concrete_3point.in

    inp = Inp_info(fileName)
    nodes, eSets = inp.nodes, inp.eSets
    body = Body(nodes=nodes, elements=list(eSets.values())[0])

    # body.show()
    body.show2d()
