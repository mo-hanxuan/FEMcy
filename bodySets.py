import sys
sys.path.append("./elementsBody_types")
from elementsBody_beam import ElementsBody_beam
from elementsBody_tetra import ElementsBody_tetra
from elementsBody_cube import ElementsBody_cube
from elementsBody_triangle import ElementsBody_triangle
from elementsBody_triPrism import ElementsBody_triPrism
from readInp import readInp
import copy
from progressBar import progressBar_percentage
import numpy as np
import multiprocessing

import mhxOpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class bodySets(object):

    """
        the integration of multiple element sets
        each set can have different types of element
    """

    mapEleType = {
        "C3D8": ElementsBody_cube,
        "C3D6": ElementsBody_triPrism,
        "3D4": ElementsBody_tetra,
        "B31": ElementsBody_beam,
        "S3": ElementsBody_triangle,
    }


    def __init__(self, nodes, eSets):
        sets = {}
        for eType in eSets:
            sets[eType] = self.mapEleType[eType](nodes, eSets[eType])
        self.nodes = nodes
        self.eSets = eSets
        self.sets = sets
    

    def get_allEdges(self):
        edgeDic = set()
        for eType in self.sets:
            if not hasattr(self.sets[eType], "edgeDic") or self.sets[eType].edgeDic == None:
                self.sets[eType].get_allEdges()
            edgeDic |= set(self.sets[eType].edgeDic.keys())
        self.edgeDic = edgeDic

    
    def get_allFacets(self):
        facetDic = set()
        for eType in self.sets:
            if not hasattr(self.sets[eType], "facetDic") or self.sets[eType].facetDic == None:
                self.sets[eType].get_allFacets()
            facetDic |= set(self.sets[eType].facetDic.keys())
        self.facetDic = facetDic
        
    
    def writeEdgesInp(self, fileName=None):
        """
            write an .inp file with beam (or edge) elements
        """
        if not hasattr(self, "edgeDic") or self.edgeDic == None:
            self.get_allEdges()
        
        if fileName == None:
            path = input("\033[40;35;1m please give the path for output file: \033[0m")
            fileName = input("\033[40;33;1m please give the .inp file name: \033[0m")
            file_ = path + "/" + fileName + "_beams.inp"
        else:
            file_ = fileName

        with open(file_, "w") as file:
            file.write("*NODE \n")
            for node in self.nodes:
                file.write("%s,  %s, %s, %s\n" % (
                    node, *self.nodes[node]
                ))
            file.write("*ELEMENT,TYPE=B31 \n")
            idx = 0
            for beam in self.edgeDic:
                idx += 1
                file.write("%s,  %s, %s\n" % (
                    idx, *beam
                ))
        print("file \033[36;1m{}\033[0m has been written.".format(file_))

    
    def getBCC(self):
        """
            converse itself the Body Center Cells
        """
        ### get the extra BCC nodes
        maxNodes = max(self.nodes)
        nodes = copy.deepcopy(self.nodes)
        idx = 0
        for eType in self.sets:
            for iele in range(len(self.sets[eType].elements)):
                idx += 1
                nodes[maxNodes + idx] = list(map(float, self.sets[eType].eleCenter(iele)))
        
        ### get all beams in BCC cells
        beamsEles = []
        idx = 0
        for eType in self.sets:
            for iele in range(len(self.sets[eType].elements)):
                idx += 1
                for nod in self.sets[eType].elements[iele]:
                    beamsEles.append([nod, maxNodes + idx])
        beamsBody = ElementsBody_beam(nodes, np.array(beamsEles))

        ### write the new .inp file with beam elements
        if input("\033[32;1m do you want to write the .inp file for BCC? (y/n): \033[0m") in ['y', '']:
            beamsBody.writeEdgesInp()

        ### compute the average beam length
        if input("\033[35;1m do you want to get the average beam length for BCC? (y/n): \033[0m") in ['y', '']:
            print("\033[32;1m the average beam length = {} \033[0m".format(
                beamsBody.averageBeamsLength()
            ))
        return beamsBody
    

    def getFCC(self):
        ### get the extra FCC nodes
        maxNodes = max(self.nodes)
        nodes = copy.deepcopy(self.nodes)
        self.get_allFacets()
        for ifacet, facet in enumerate(self.facetDic):
            nodesCoo = np.array([self.nodes[nod] for nod in facet])
            nodesCoo = np.einsum("ij -> j", nodesCoo) / len(nodesCoo)
            nodes[maxNodes + ifacet + 1] = list(map(float, nodesCoo))

        ### get all beams in FCC cells
        beamsEles = []
        for ifacet, facet in enumerate(self.facetDic):
            for nod in facet:
                beamsEles.append([nod, maxNodes + ifacet + 1])
        beamsBody = ElementsBody_beam(nodes, np.array(beamsEles))

        ### write the new .inp file with beam elements
        if input("\033[35;1m do you want to write the .inp file? (y/n): \033[0m") in ['y', '']:
            beamsBody.writeEdgesInp()

        ### compute the average beam length
        if input("\033[35;1m do you want to get the average beam length? (y/n): \033[0m") in ['y', '']:
            print("\033[32;1m the average beam length = {} \033[0m".format(
                beamsBody.averageBeamsLength()
            ))
        return beamsBody
    

    def getOctetTruss(self):
        ### get the extra FCC nodes
        maxNodes = max(self.nodes)
        nodes = copy.deepcopy(self.nodes)
        self.get_allFacets()
        fromFacetTupleToNodeNumber = {}
        for ifacet, facet in enumerate(self.facetDic):
            nodesCoo = np.array([self.nodes[nod] for nod in facet])
            nodesCoo = np.einsum("ij -> j", nodesCoo) / len(nodesCoo)
            nodes[maxNodes + ifacet + 1] = list(map(float, nodesCoo))
            fromFacetTupleToNodeNumber[facet] = maxNodes + ifacet + 1

        ### get all beams in FCC cells
        beamsEles = []
        for ifacet, facet in enumerate(self.facetDic):
            for nod in facet:
                beamsEles.append([nod, maxNodes + ifacet + 1])

        ### get regular octahedron beams in all cubic cells
        for eType in self.sets:
            for iele, ele in enumerate(self.sets[eType].elements):
                nod = []
                for ifacet, facet1 in enumerate(self.sets[eType].eleFacet[iele]):
                    for jfacet in range(ifacet + 1, len(self.sets[eType].eleFacet[iele])):
                        facet2 = self.sets[eType].eleFacet[iele][jfacet]
                        if len(set(facet1) & set(facet2)) == 2:
                            nod1 = fromFacetTupleToNodeNumber[facet1]
                            nod2 = fromFacetTupleToNodeNumber[facet2]
                            beamsEles.append([nod1, nod2])

        ### get the body of beams
        beamsBody = ElementsBody_beam(nodes, np.array(beamsEles))

        ### write the new .inp file with beam elements
        if input("\033[35;1m do you want to write the .inp file? (y/n): \033[0m") in ['y', '']:
            beamsBody.writeEdgesInp()

        ### compute the average beam length
        if input("\033[35;1m do you want to get the average beam length? (y/n): \033[0m") in ['y', '']:
            print("\033[32;1m the average beam length = {} \033[0m".format(
                beamsBody.averageBeamsLength()
            ))
        return beamsBody
    

    def getDiamond(self, ):
        cubesBody = ElementsBody_cube(self.nodes, self.eSets["C3D8"])
        beamsBody = conformalLattice(cubesBody, "cell_library/diamond.yml")
        return beamsBody
    

    def writePartitionSets(self, fileName=None):
        self.get_allEdges()
        beamsBody = ElementsBody_beam(self.nodes, np.array(list(self.edgeDic)))
        beamsBody.writePartitionSets(fileName=fileName)
    

    def getXyzRange(self):
        nodes = np.array([self.nodes[_] for _ in self.nodes])
        minX = nodes[:, 0].min()
        maxX = nodes[:, 0].max()
        minY = nodes[:, 1].min()
        maxY = nodes[:, 1].max()
        minZ = nodes[:, 2].min()
        maxZ = nodes[:, 2].max()
        ranges = ((minX, maxX), (minY, maxY), (minZ, maxZ))
        biggestRange = max(maxX - minX, maxY - minY, maxZ - minZ)
        regionCen = ((minX + maxX) / 2., (minY + maxY) / 2., (minZ + maxZ) / 2.)
        return ranges, biggestRange, regionCen
    
  
    def visualizeEdges(self, edges, regionCen, stretchRatio):
        glLineWidth(2.)
        red, green, blue = 0.0, 0.8, 0.0
        glColor4f(red, green, blue, 1.0)
        glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
        glBegin(GL_LINES)
        for edge in edges:
            glVertex3f(
                (self.nodes[edge[0]][0] - regionCen[0]) * stretchRatio, 
                (self.nodes[edge[0]][1] - regionCen[1]) * stretchRatio, 
                (self.nodes[edge[0]][2] - regionCen[2]) * stretchRatio
            )
            glVertex3f(
                (self.nodes[edge[1]][0] - regionCen[0]) * stretchRatio, 
                (self.nodes[edge[1]][1] - regionCen[1]) * stretchRatio, 
                (self.nodes[edge[1]][2] - regionCen[2]) * stretchRatio
            )
        glEnd()
        ### ---------------------------------------------------------------
        glutSwapBuffers()  # swap the buffer and show the view


if __name__ == "__main__":
    fileName = input("\033[40;33;1m{}\033[0m".format(
        "please input the .inp file name (include the path): "
    ))
    job = fileName.split('/') if '/' in fileName else fileName.split('\\')
    job = job[-1].split('.')[0]

    obj1 = bodySets(*readInp(fileName=fileName))

    scale = float(input("\033[35;1m Scale up the whole structure. scale = \033[0m"))
    for node in obj1.nodes:
        obj1.nodes[node] = [obj1.nodes[node][_] * scale for _ in range(3)]

    if input("\033[35;1m do you want to write the beams .inp file? (y/n): \033[0m") in ['y', '']:
        obj1.writeEdgesInp(fileName=job)

    ### get the average beam length
    if input("\033[35;1m do you want to get the average beam length? (y/n): \033[0m") in ['y', '']:
        if not hasattr(obj1, "allEdges") or obj1.allEdges == None:
            obj1.get_allEdges()
        elements = np.array(list(obj1.edgeDic))
        newbody = ElementsBody_beam(obj1.nodes, elements)
        print("\033[32;1m the average beam length = {} \033[0m".format(
            newbody.averageBeamsLength()
        ))
    
    ### get BCC, FCC, or Octet-truss
    print("\033[40;33;1m do you want to transverse the body to other kinds of cells? \033[0m")
    inputWords = input("\033[35;1m iuput a number, 0: NO, 1: BCC, 2: FCC, 3: Octet-truss 4: Diamond \n\033[0m")
    if inputWords in ['1', '2', '3', '4']:
        instructions = {
            '1': obj1.getBCC,
            '2': obj1.getFCC,
            '3': obj1.getOctetTruss,
            '4': obj1.getDiamond,
        }
        beamsBody = instructions[inputWords]()
        if input("\033[35;1m write it to a file? (y/n): \033[0m") in ["y", ""]:
            beamsBody.writeEdgesInp()
    
        ### use the new structure to replace the original structure?
        options = {'1': "BCC", '2':"FCC", '3':"Octet-truss", '4':"Diamond"}
        if input(
            "\033[35;1m {} \033[40;33;1m{}\033[35;1m {} \033[0m".format(
                "do you want to use the new structure of",
                options[inputWords],
                "to replace the original structure? (y/n):"
            )
        ) in ["y", ""]:
            obj1 = bodySets(nodes=beamsBody.nodes, eSets={"B31": beamsBody.elements})
