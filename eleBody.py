from readInp import readInp
import torch as tch
import numpy as np
import time
from progressBar import *


class EleBody(object):
    """
        author: mohanxuan,  mo-hanxuan@sjtu.edu.cn
        license: Apache 2.0
    """
    def __init__(self, nodes, elements, name='elesBody1'):
        """
        nodes[] are coordinates of all nodes
        elements[] are the corresponding node number for each element 
        """
        if isinstance(elements, dict):
            ### if input elements is a dict with 
            ### key: string of elementType, values: elements of this element type
            elements = list(elements.values())[0]
        for node in nodes:
            if (not tch.is_tensor(nodes[node])) and type(nodes[node]) not in [type([]), type(np.array([0.]))]:
                print('type(nodes[node]) =', type(nodes[node]))
                raise ValueError('item in nodes should be a torch tensor or a list or an numpy array')
            break
        if not tch.is_tensor(elements):
            raise ValueError('elements should be a torch tensor')
        for node in nodes:
            if tch.is_tensor(nodes[node]):
                print('nodes[node].size() =', nodes[node].size())
                if nodes[node].size()[0] != 3:
                    raise ValueError('nodes coordinates should 3d')
            else:
                # print('len(nodes[node]) =', len(nodes[node]))
                if len(nodes[node]) != 3:
                    raise ValueError('nodes coordinates should 3d')
            break
        if len(elements.size()) != 2:
            raise ValueError('len(elements.size()) should be 2 !')
        if elements.max() > max(nodes):
            print('elements.max() =', elements.max())
            print('max(nodes) =', max(nodes))
            raise ValueError('maximum element nodes number > max nodes number')
        
        self.nodes = nodes
        self.elements = elements
        self.name = name

        self.nod_ele = None
        self.eleNeighbor = None
        self.allFacets = None  # all element facets of this body
        self.eCen = {}  # elements center
        self._celent = {}  # chosen element length

        # node number starts from 1
        # element number starts from 0


    def get_nod_ele(self):  # node number -> element number
        if not self.nod_ele:
            nod_ele = {i:set() for i in self.nodes}
            for iele, ele in enumerate(self.elements):
                for node in ele:
                    nod_ele[int(node)].add(iele)
            self.nod_ele = nod_ele

        return self.nod_ele


    def get_eleNeighbor(self):  # get the neighbor elements of the given element
        if not self.nod_ele:
            self.get_nod_ele()
        if not self.eleNeighbor:
            neighbor = {i: set() for i in range(len(self.elements))}
            for iele, ele in enumerate(self.elements):
                for node in ele:
                    for eNei in self.nod_ele[int(node)]:
                        if eNei != iele:
                            neighbor[iele].add(eNei)
            self.eleNeighbor = neighbor
        return self.eleNeighbor
    

    def eleCenter(self, iele):
        if iele in self.eCen:
            return self.eCen[iele]
        else:
            nodes = [self.nodes[int(j)] for j in self.elements[iele]]
            nodes = tch.tensor(nodes)
            self.eCen[iele] = [
                nodes[:, 0].sum() / len(nodes), 
                nodes[:, 1].sum() / len(nodes), 
                nodes[:, 2].sum() / len(nodes), 
            ]
            return self.eCen[iele]


    def findHorizon(self, iele, inHorizon):
        """
            find horizon elements for a given element
                (horizon means a specific range around the given element)
            iuput:
                iele: int, id of the given element 
                inHorizon: function, judge whether ele is inside horizon 
                    (generally by the ele's center xyz)
            output:
                horizon: list, the element number list of the horizon
            methods:
                use BFS to serach neighbor elements layer by layer
                (which prevent searching all elements from whold body)
            note:
                elements id starts from 0
            author: mo-hanxuan@sjtu.edu.cn
            LICENSE: Apache license
        """
        
        ### preprocess of data preparation, 
        ### automatically executes at the 1st call of this function
        ###     inorder to save time for many following calls
        if (not hasattr(self, "eleNeighbor")) or self.eleNeighbor == None:
            self.get_eleNeighbor()
        if (not hasattr(self, "eLen")) or self.eLen == None:
            self.get_eLen(mod="average")
        if len(self.eCen) != len(self.elements):  # prepare self.eCen for all elements
            for idx in range(len(self.elements)):
                self.eleCenter(idx)
        
        horizon = {iele}
        lis = [iele]
        while lis:
            lisNew = []
            for ele in lis:
                for nex in self.eleNeighbor[ele]:
                    if nex not in horizon:
                        if inHorizon(iele, nex):
                            lisNew.append(nex)
                            horizon.add(nex)
            lis = lisNew
        return horizon


    def inHorizon(self, iele, ele):
        """
            judge whether ele is in the horizon of iele
            horizon has center at iele
                and has length of 11 * eLen at all 3 directions
            input:
                iele: int, the center element id of horizon
                ele: int, element id to be judged whether inside the horizon
            output:
                variable of True or False
            note:
                use self.eCen instead of self.eleCenter(),
                because each time call eleCenter() consumes an 'if' operation
                which slows down the process if we need centers frequently
        """
        eCen = self.eCen
        eps = 0.01  # relative error of eLen
        lenRange = (5. + eps) * self.eLen
        if abs(eCen[ele][0] - eCen[iele][0]) <= lenRange:
            if abs(eCen[ele][1] - eCen[iele][1]) <= lenRange:
                if abs(eCen[ele][2] - eCen[iele][2]) <= lenRange:
                    return True
        return False
    

    def ratioOfVisualize(self):
        """
            get the retio of visualization for this body
        """
        # get the maximum x and minimum 
        beg = time.time()
        xMax = max(self.nodes.values(), key=lambda x: x[0])[0]
        xMin = min(self.nodes.values(), key=lambda x: x[0])[0]
        yMax = max(self.nodes.values(), key=lambda x: x[1])[1]
        yMin = min(self.nodes.values(), key=lambda x: x[1])[1]
        zMax = max(self.nodes.values(), key=lambda x: x[2])[2]
        zMin = min(self.nodes.values(), key=lambda x: x[2])[2]
        print(
            "\033[40;35m{} \033[40;33m{}, \033[40;35m{} \033[40;33m{}\n"
            "\033[40;35m{} \033[40;33m{}, \033[40;35m{} \033[40;33m{}\n"
            "\033[40;35m{} \033[40;33m{}, \033[40;35m{} \033[40;33m{}\033[0m".format(
                "xMax =", xMax, "xMin =", xMin,
                "yMax =", yMax, "yMin =", yMin,
                "zMax =", zMax, "zMin =", zMin,
            )
        )
        print("time for max min computing =", time.time() - beg)
        self.maxLen = max(xMax - xMin, yMax - yMin, zMax - zMin)
        
        self.ratio_draw = 1. if self.maxLen < 10. else 5. / self.maxLen
        print("\033[35;1m{} \033[40;33;1m{}\033[0m".format(
            "this body's ratio of visualization (self.ratio_draw) =", self.ratio_draw
        ))
        return self.ratio_draw


    def ele_directional_range(self, iele, direction=[0., 0., 1.]):
        """
            get the elements min or max coordiantes along a direction
            aim for computation of twin length
            return:
                minVal, maxVal
            author: mo-hanxuan@sjtu.edu.cn
            LICENSE: Apache license
        """
        ### unitize the direction vector
        eps = 1.e-6  # a value nearly 0
        drc = tch.tensor(direction)
        drc_len = (drc**2).sum() ** 0.5  # vector length of direction
        if drc_len < eps:
            raise ValueError(
                "input direction vector is almost 0, "
                "can not comput its length neumerically"
            )
        drc = drc / drc_len  # unitize

        def drcMinMax(id, drc):
            ### compute directional min and max value of an element,
            ### where input dic must be unitized
            nodesCoo = tch.tensor(
                [self.nodes[int(i)] for i in self.elements[id]]
            )
            minVal = min(nodesCoo, key=lambda xyz: xyz @ drc) @ drc
            maxVal = max(nodesCoo, key=lambda xyz: xyz @ drc) @ drc
            return float(minVal), float(maxVal)

        if isinstance(iele, int):
            return drcMinMax(iele, drc)
        elif iele == "all":
            minVals, maxVals = [], []
            for i in range(len(self.elements)):
                if i % 100 == 0:
                    progressBar_percentage(i / len(self.elements) * 100.)
                min_, max_ = drcMinMax(i, drc)
                minVals.append(min_)
                maxVals.append(max_)
            print("")  # break line for progress bar
            return minVals, maxVals
        else:
            raise ValueError("iele must be an int, or string 'all' to include all elements")
