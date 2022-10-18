"""some user defined subroutines, 
   which can be specified by the user"""
import taichi as ti


@ti.kernel
def user_dirichletBC(
                    dof: ti.template(),  # variables to be specified
                    nodeSet: ti.template(), 
                    dm: int,  # dm is spatial dimension
                    dm_specified: int,  # the specified dimendion of dirichlet BC 
                    nodes: ti.template(), 
                    time: float,  # specific value of dirichlet boundary condition
                    ):
    """ user defined Dirichlet BC 
        you can specify self.dof in this function"""
    pi = 3.141592653589793
    center = ti.Vector([40., 5., 0.])
    for node_ in nodeSet:
        node = nodeSet[node_]
        i_global = node * dm + dm_specified
        angle = time * pi
        rota = ti.Matrix([ 
            [ti.cos(angle), ti.sin(angle), 0.], 
            [-ti.sin(angle), ti.cos(angle), 0.], 
            [0., 0., 1.], 
        ])
        new_x = rota @ (nodes[node] - center) + center
        disp = new_x - nodes[node]
        dof[i_global] = disp[dm_specified]