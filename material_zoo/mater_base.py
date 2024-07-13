"""define an abstract interface class of materials, 
so that many other kinds of materials can be derived from this class
"""

import abc
import taichi as ti


class MaterBase(abc.ABC):

    @abc.abstractmethod
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @abc.abstractmethod
    def constitutiveOfSmallDeform(self, deform_grad: ti.template(), cauchy_stress: ti.template(),
                                  ddsdde: ti.template()):
        pass

    @abc.abstractmethod
    def constitutiveOfLargeDeform(self, deform_grad: ti.template(), cauchy_stress: ti.template(),
                                  ddsdde: ti.template()):
        pass

    @abc.abstractmethod
    def elasticEnergyDensity(self, deform_grad: ti.template()):
        pass
