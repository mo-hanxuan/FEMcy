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

    @abc.abstractmethod
    def shapeFunc(self, natural_coo: ti.template()):
        pass

    @abc.abstractmethod
    def dshape_dnat(self, natural_coo: ti.template()):
        pass

    @abc.abstractmethod
    def shapeFunc_pyscope(self, natural_coo: np.ndarray):
        pass

    @abc.abstractmethod
    def dshape_dnat_pyscope(self, natural_coo: np.ndarray):
        pass

    @abc.abstractmethod
    def globalNormal(self, nodes: np.ndarray, facet: list, integPointId=0):
        pass

    @abc.abstractmethod
    def strainMtrx(self, dsdx):
        pass

    @abc.abstractmethod
    def getMesh(self, elements: np.ndarray):
        pass

    @abc.abstractmethod
    def extrapolate(self, internal_vals: ti.template(), nodal_vals: ti.template()):
        pass
