"""a base abstract class of input info from file
the input info includes nodes, elements, materials, boundary-conditions, etc.
"""

import abc


class InpInfoBase(abc.ABC):

    @abc.abstractmethod
    def __init__(self, file_name: str):
        pass

    @abc.abstractmethod
    def read_node_element(self, file_name: str):
        pass

    @abc.abstractmethod
    def read_set(self, file_name: str):  # read input-file to get node-set and element-set
        pass

    @abc.abstractmethod
    def read_face_set(self, file_name: str):
        pass

    @abc.abstractmethod
    def get_boundary_condition(self, file_name: str):
        pass

    @abc.abstractmethod
    def read_material(self, file_name: str):
        pass

    @abc.abstractmethod
    def read_geometric_nonlinear(self, file_name: str):
        pass

    @abc.abstractmethod
    def read_time_inc(self, file_name: str):  # read the time increment
        pass
