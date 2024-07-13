"""define an abstract interface class of elements, 
so that many other kinds of elements can be derived from this class
"""
import abc
import numpy as np
import taichi as ti


class ElementBase(abc.ABC):

    @abc.abstractmethod
    def __init__(self, ):
        pass

    # input the natural coordinate and get the shape function values
    @abc.abstractmethod
    def shapeFunc(self, natural_coo: ti.template()):
        pass

    # derivative of shape function with respect to natural coodinate
    @abc.abstractmethod
    def dshape_dnat(self, natural_coo: ti.template()):
        pass

    # input the natural coordinate and get the shape function values in pyscope
    @abc.abstractmethod
    def shapeFunc_pyscope(self, natural_coo: np.ndarray):
        pass

    # derivative of shape function with respect to natural coodinate in pyscope
    @abc.abstractmethod
    def dshape_dnat_pyscope(self, natural_coo: np.ndarray):
        pass

    # deduce the normal vector in global coordinate for a given facet
    @abc.abstractmethod
    def globalNormal(self, nodes: np.ndarray, facet: list, integPointId=0):
        pass

    # strain for the stiffness matrix
    @abc.abstractmethod
    def strainMtrx(self, dsdx):
        pass

    # get the elements and outer surface of the mesh
    @abc.abstractmethod
    def getMesh(self, elements: np.ndarray):
        pass

    # extrapolate the internal Gauss points' vals to the nodal vals
    @abc.abstractmethod
    def extrapolate(self, internal_vals: ti.template(), nodal_vals: ti.template()):
        pass
