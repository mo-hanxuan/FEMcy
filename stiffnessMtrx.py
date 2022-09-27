"""
construct the stiffness matrix
"""
import taichi as ti
import numpy as np
import time; from typing import Tuple, Union
from body import Body
from readInp import *
from material import *
from element_linear_triangular import Element_linear_triangular
from element_quadratic_triangular import Element_quadratic_triangular
from element_linear_tetrahedral import Element_linear_tetrahedral
from conjugateGradientSolver import ConjugateGradientSolver_rowMajor as CG
from tiMath import a_equals_b_plus_c_mul_d, a_from_b, c_equals_a_minus_b, field_multiply, field_norm, get_index_ti, vec_mul_voigtMtrx, field_abs_max


@ti.data_oriented
class System_of_equations:
    """
    the system of equations that applies to solve the linear equation systems (can be modified to nonlinear system)
    including:
        sparseMtrx: the sparse stiffness matrix utilized to solve the dofs
        rhs: right hand side of the equation system
    """
    def __init__(self, body: Body, material, geometric_nonlinear: bool):
        self.dm = body.dm  # spatial dimension of the system
        self.geometric_nonlinear = geometric_nonlinear
        self.body = body
        self.elements, self.nodes = body.elements, body.nodes 
        
        self.sparseMtrx = ti.field(ti.f64)
        sparseMtrx_components = ti.root.pointer(ti.ij, (body.nodes.shape[0] * body.dm, body.nodes.shape[0] * body.dm))
        sparseMtrx_components.place(self.sparseMtrx)
        
        ### sparseMtrx @ dof = rhs
        self.rhs = ti.field(ti.f64, shape=(body.nodes.shape[0] * body.dm, ))  # right hand side of the equation system
        self.dof = ti.field(ti.f64, shape=(body.nodes.shape[0] * body.dm, ))  # degree of freedom that needs to be solved
        
        ### define the element types of the body, must be modified latter!!!
        if body.np_elements.shape[1] == 3:
            self.ELE = Element_linear_triangular()
        elif body.np_elements.shape[1] == 6:
            self.ELE = Element_quadratic_triangular()
        elif body.np_elements.shape[1] == 4:
            self.ELE = Element_linear_tetrahedral()
        elif body.np_elements.shape[1] == 10:
            self.ELE = Element_quadratic_tetrahedral()

        ### deformation gradient
        self.F = ti.Matrix.field(self.dm, self.dm, ti.f64, 
                                shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))
        
        ### stress and strain (infinitesimal strain for small deformation and Green strain for large deformation)
        self.cauchy_stress = ti.Matrix.field(self.dm, self.dm, ti.f64, 
                        shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))  
        self.strain = ti.Matrix.field(self.dm, self.dm, ti.f64, 
                        shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0])) 
        self.mises_stress = ti.field(ti.f64, 
                        shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))   

        ### variables related to geometric nonlinear                              
        self.nodal_force = ti.field(ti.f64, shape=(self.dm * self.body.nodes.shape[0]))  
        self.residual_nodal_force = ti.field(ti.f64, shape=(body.nodes.shape[0] * body.dm, ))

        ### dsdx (derivative of shape functio with respect to current coordinate), and volume of each guass point
        self.dsdx = ti.Matrix.field(self.elements[0].n, self.dm, ti.f64, 
                                    shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))
        self.vol = ti.field(ti.f64, shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))
        
        ### constitutive material (e.g., elastic constants) of each element
        self.material = material; self.C = material.C
        self.ddsdde = ti.Matrix.field(n=material.C.n, m=material.C.m, dtype=ti.f64, 
                                      shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))
        self.ddsdde_init()
        
        ### link a node to the related elements
        body.get_nodeEles()
        maxLen = max(len(eles) for eles in body.nodeEles)
        self.nodeEles = ti.Vector.field(maxLen, ti.i32, shape=(self.nodes.shape[0], ))
        nodeEles = -np.ones((body.nodes.shape[0], maxLen))
        for node in range(len(nodeEles)):
            nodeEles[node, 0:len(body.nodeEles[node])] = body.nodeEles[node][:]
        self.nodeEles.from_numpy(nodeEles)

        ### get the sparseIJ (index 0 of each row stores the number of effective indexes in this row)
        body.get_coElement_nodes()
        maxLen = max(len(body.coElement_nodes[node]) for node in range(body.nodes.shape[0]))
        self.sparseIJ = ti.Vector.field(maxLen * self.dm + 1, ti.i32, shape=(self.nodes.shape[0] * self.dm, ))
        sparseIJ = -np.ones((self.nodes.shape[0] * self.dm, maxLen * self.dm + 1))
        for node0 in range(self.nodes.shape[0]):
            js = [node1 * self.dm + i for node1 in body.coElement_nodes[node0] for i in range(self.dm)]
            for i in range(self.dm):
                sparseIJ[node0 * self.dm + i][1:len(js)+1] = js[:]
                sparseIJ[node0 * self.dm + i][0] = len(js)
        self.sparseIJ.from_numpy(sparseIJ)

        ### init the row major form of sparse matrix
        print("\033[32;1m shape of the sparseIJ is {} \033[0m".format(self.sparseIJ.shape))
        self.sparseMtrx_rowMajor = ti.Vector.field(self.sparseIJ[0].n - 1, ti.f64, shape=(self.sparseIJ.shape[0], ))

        ### initial variables related to time increments
        self.time0 = 0.; self.time1 = 0.
        self.dt = 0.
        self.dof_old = ti.field(ti.f64, shape=(body.nodes.shape[0] * body.dm, ))  # degree of freedom at last time step

        ### some variables for print
        self.compiled = False  # indicate whether the assemble_sparseMtrx has been compiled
        self.visualize_field = ti.field(ti.f64,  # a field for visualization, you can visualize some idexes of stress or strain
                        shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))   
        self.nodal_vals = ti.Vector.field(self.elements[0].n, ti.f64, shape=(self.elements.shape[0],)) # visualization of nodal strain or stress


    @ti.kernel
    def ddsdde_init(self, ):
        """get the ddsdde at each integration point of each element"""
        for ele, igp in self.ddsdde:
            self.ddsdde[ele, igp] = self.C
    

    @ti.kernel
    def assemble_sparseMtrx(self, ):
        dm, sparseMtrx, sparseMtrx_rowMajor, \
        nodeEles, nodes, elements, ddsdde  = ti.static(
            self.body.dm, self.sparseMtrx, self.sparseMtrx_rowMajor, \
            self.nodeEles, self.nodes, self.elements, self.ddsdde)
        ### refresh the sparse Matrix if it has been assemble in the previous increment
        for i, j in sparseMtrx:
            sparseMtrx[i, j] = 0.
        for node0 in nodes:
            for iele in range(nodeEles[0].n):
                if nodeEles[node0][iele] != -1:
                    ele = nodeEles[node0][iele]

                    ### get the sequence of this node in the element
                    nid = get_index_ti(elements[ele], node0)
                    if nid == -1:
                        print("\033[31;1m Error, index not found. nid = -1 \033[0m")

                    ### get the gradient of u, express it as the coefficients of u of different nodes
                    for igp in range(self.ELE.gaussPoints.shape[0]):
                        dsdx = self.dsdx[ele, igp]
                        
                        ### length of each component = dm * nodes_of_element
                        strain = self.ELE.strain_for_stiffnessMtrx_taichi(dsdx)
                        ### C : ε， each component is a vector with dimension s, thus maybe can not use operation @ directly
                        stress_voigt = ddsdde[ele, igp] @ strain

                        ### dsdx mutiplies the stress
                        dsdx_x_stress = vec_mul_voigtMtrx(dsdx[nid, :], stress_voigt)
                        
                        ### get the volume related to this Gauss point
                        vol = self.vol[ele, igp]

                        ### integrate to the large sparse matrix
                        Is = ti.Vector([node0 * dm + i for i in range(dm)])
                        Js_ = elements[ele] * dm  ## need to be modified here
                        Js = ti.Vector([x + i for x in Js_ for i in range(dm)])
                        for i_local in range(dm):
                            i_global = Is[i_local]
                            for j_local in range(Js.n):
                                j_global = Js[j_local]
                                sparseMtrx[i_global, j_global] = \
                                sparseMtrx[i_global, j_global] + dsdx_x_stress[i_local, j_local] * vol
        ### transform the sparse matrix into row major
        for i in sparseMtrx_rowMajor:
            for j in range(sparseMtrx_rowMajor[0].n):
                sparseMtrx_rowMajor[i][j] = 0.
        for i, j in sparseMtrx:
            j0 = self.sparseMatrix_get_j(i, j)
            sparseMtrx_rowMajor[i][j0] = sparseMtrx[i, j]
  

    @ti.kernel
    def dirichletBC_linearEquations(self, 
                    nodeSet: ti.template(), dm_specified: int,  # the specified dimendion of dirichlet BC 
                    sval: float,  # specific value of dirichlet boundary condition
                    ):
        """apply dirichlet boundary condition to the body
           modify the sparse matrix and the rhs (right hand side)
           this is for linear equation systems"""
        ### impose dirichlet BC at specific nodes
        for node_ in nodeSet:
            node = nodeSet[node_]
            i_global = node * self.dm + dm_specified

            ### modify the right hand side
            for j0 in range(self.sparseIJ[i_global][0]):
                j_global = self.sparseIJ[i_global][j0 + 1]
                ### use the symmetric property of the sparse matrix, find sparseMtrx[j_global, i_global]
                i0 = self.sparseMatrix_get_j(j_global, i_global)
                self.rhs[j_global] = self.rhs[j_global] - sval * self.sparseMtrx_rowMajor[j_global][i0]
            self.rhs[i_global] = sval

            ### modify the sparse matrix
            for j0 in range(self.sparseIJ[i_global][0]):
                j_global = self.sparseIJ[i_global][j0 + 1]
                self.sparseMtrx_rowMajor[i_global][j0] = 0.
                i0 = self.sparseMatrix_get_j(j_global, i_global)
                self.sparseMtrx_rowMajor[j_global][i0] = 0.
            i0 = self.sparseMatrix_get_j(i_global, i_global)
            self.sparseMtrx_rowMajor[i_global][i0] = 1.


    def dirichletBC_forNewtonMethod(self, dirichletBCs):
        for dirichletBC in dirichletBCs:
            self.dirichletBC_forNewtonMethod_kernel(nodeSet=dirichletBC["node_set"], 
                                                    dm_specified=dirichletBC["dof"],
                                                    sval=dirichletBC["val"])
    @ti.kernel
    def dirichletBC_forNewtonMethod_kernel(self, 
                    nodeSet: ti.template(), dm_specified: int,  # the specified dimendion of dirichlet BC 
                    sval: float,  # specific value of dirichlet boundary condition
                    ):
        """apply dirichlet boundary condition to the body
           modify the sparse matrix and the residual force
           this is for Newton method, ref: https://scorec.rpi.edu/~granzb/notes/dbcs/dbcs.pdf """
        
        ## impose dirichlet BC at dof
        for node_ in nodeSet:
            node = nodeSet[node_]
            i_global = node * self.dm + dm_specified
            self.dof[i_global] = sval
        
        ### impose dirichlet BC at residual force and sparse matrix
        for node_ in nodeSet:
            node = nodeSet[node_]
            i_global = node * self.dm + dm_specified

            ### modify the residual force
            self.residual_nodal_force[i_global] = 0.

            ### modify the sparse matrix (i.e., the Jacobian)
            for j0 in range(self.sparseIJ[i_global][0]):
                j_global = self.sparseIJ[i_global][j0 + 1]
                self.sparseMtrx_rowMajor[i_global][j0] = 0.
                i0 = self.sparseMatrix_get_j(j_global, i_global)
                self.sparseMtrx_rowMajor[j_global][i0] = 0.
            i0 = self.sparseMatrix_get_j(i_global, i_global)
            self.sparseMtrx_rowMajor[i_global][i0] = 1.
    

    @ti.kernel
    def dirichletBC_dof(self, 
                    nodeSet: ti.template(), dm_specified: int,  # the specified dimendion of dirichlet BC 
                    sval: float,  # specific value of dirichlet boundary condition
                    ):
        ### impose dirichlet BC at dof
        for node_ in nodeSet:
            node = nodeSet[node_]
            i_global = node * self.dm + dm_specified
            self.dof[i_global] = sval
    

    def neumannBC(self, load_facets, load_val: float, load_dir=np.array([])):  # Neumann boundary condition, 
                                                                       # should be modified latter to taichi version!!!
        """
        load_facets: list(tuple), the boundary facets with load (i.e., the boundary with specific flux value)
        freeload_facets: boundary without specified value of flux(field gradient)

            if no load in the surface, flux term doesn't need to be computed, because:
                1.  free surface, with no traction force on the surface, thus the flux term is 0
                2.  surface with Dirichlet BC, nodal displacement is specified, 
                    flux term is not 0, but can be automatically deduced from Dirichlet BC
                    (technically, this term doesn't need to be computed, 
                    because after zero-one setting method for DirichletBC, the effort here will be erased) 
        """
        body = self.body; ELE = self.ELE
        body.get_boundary()
        self.rhs.fill(0.)  # refresh the right hand side before apply Neumann BC
        
        for facet in load_facets:
            ele = body.boundary[facet]
            for node0 in facet:  # traction force of this facet applies to node0

                ### obtain the facet normals on the free-load boundary 
                ###     (points to the outside of the element)
                localNodes = np.array([body.np_nodes[node, :] for node in body.np_elements[ele, :]])
                eleNodesList = body.np_elements[ele, :].tolist()
                localFacet = [eleNodesList.index(i) for i in facet]
                for gaussId in range(ELE.gaussPointNum_eachFacet):
                    normal_vector, area_x_gaussWeight = ELE.global_normal(nodes=localNodes, 
                                                                            facet=localFacet, 
                                                                            gaussPointId=gaussId)
                    ### get the flux, which can also be interpreted as traction force
                    if len(load_dir) == 0: 
                        flux = load_val * normal_vector * area_x_gaussWeight
                    else:
                        flux = load_val * load_dir * area_x_gaussWeight   

                    ### get the sequence of this node in the element
                    natCoo = ELE.facet_natural_coos[tuple(sorted(localFacet))][gaussId]
                    nid = list(body.np_elements[ele, :]).index(node0)
                    shapeVal = ELE.shapeFunc_pyscope(natCoo)[nid] 

                    for i in range(self.dm):
                        self.rhs[node0*self.dm + i] += flux[i] * shapeVal

    
    @ti.func
    def sparseMatrix_get_j(self, i_global, j_global):
        j_local = 0
        for j in range(self.sparseIJ[i_global][0]):
            if self.sparseIJ[i_global][j + 1] == j_global:
                j_local = j
        return j_local
        

    @ti.kernel
    def count_components_of_sparseMtrx(self, ) -> float:
        count = 0
        for i, j in self.sparseMtrx:
            count += 1
        return count
    

    def check_sparseIJ(self, ):
        """check whether indexes repeatly appear in sparseIJ"""
        for i in range(self.sparseIJ.shape[0]):
            js = set()
            for j0 in range(self.sparseIJ[i][0]):
                j = self.sparseIJ[i][j0 + 1]
                if j != -1:
                    if j in js:
                        print("\033[31;1m error, {}, {} appear repeatly \033[0m".format(i, j))
                    else:
                        js.add(j)
    

    def solve_dof(self):
        if not hasattr (self, "PCG"):
            if not self.geometric_nonlinear:
                self.PCG = CG(spm=self.sparseMtrx_rowMajor, sparseIJ=self.sparseIJ, b=self.rhs)
            else:
                self.PCG = CG(spm=self.sparseMtrx_rowMajor, sparseIJ=self.sparseIJ, b=self.residual_nodal_force)
        self.PCG.re_init() 
        self.PCG.solve()

        if not self.geometric_nonlinear:
            self.dof = self.PCG.x
        else:
            ### self.dof = self.dof - solver.x (in Newton's method)
            c_equals_a_minus_b(self.dof, self.dof, self.PCG.x)
        
        return self.PCG


    def compute_strain_stress(self, ):
        ### get the strain
        self.get_deformation_gradient()
        if not self.geometric_nonlinear:
            self.get_strain_smallDeformation()
        else:
            self.get_strain_largeDeformation()
        ### get the stress
        if not self.geometric_nonlinear:
            if isinstance(self.material, Linear_isotropic_planeStrain):
                self.constitutive_planeStrain_linear(self.ELE.gaussPoints)
            elif isinstance(self.material, Linear_isotropic_planeStress):
                self.constitutive_planeStress_linear(self.ELE.gaussPoints)
            elif isinstance(self.material, Linear_isotropic):
                self.constitutive_infinitesimal(self.ELE.gaussPoints)
        else:
            pass  # stress has been computed for geometric nonlinear case
        ### compute mises stress
        if isinstance(self.material, Linear_isotropic_planeStrain):
            self.get_mises_stress_planeStrain()
        elif isinstance(self.material, Linear_isotropic_planeStress):
            self.get_mises_stress_planeStress()
        else:
            self.get_mises_stress_3d()
    

    @ti.kernel 
    def get_mises_stress_planeStress(self, ):
        eye = ti.Matrix([[1., 0., 0.], 
                         [0., 1., 0.], 
                         [0., 0., 1.]])
        for ele in self.elements:
            for igp in range(self.ELE.gaussPoints.shape[0]):
                stress_2d = self.cauchy_stress[ele, igp]
                stress = ti.Matrix([ 
                    [stress_2d[0, 0], stress_2d[0, 1], 0.], 
                    [stress_2d[1, 0], stress_2d[1, 1], 0.],
                    [0., 0., 0.],
                ])
                deviatoric_stress = stress - eye * stress.trace() / 3.
                self.mises_stress[ele, igp] = (3./2. * (deviatoric_stress * deviatoric_stress).sum())**0.5
    

    @ti.kernel 
    def get_mises_stress_planeStrain(self, ):
        nu = self.material.poisson_ratio
        eye = ti.Matrix([[1., 0., 0.], 
                         [0., 1., 0.], 
                         [0., 0., 1.]])
        for ele in self.elements:
            for igp in range(self.ELE.gaussPoints.shape[0]):
                stress_2d = self.cauchy_stress[ele, igp]
                stress = ti.Matrix([ 
                    [stress_2d[0, 0], stress_2d[0, 1], 0.], 
                    [stress_2d[1, 0], stress_2d[1, 1], 0.],
                    [0., 0., nu * (stress_2d[0, 0] + stress_2d[1, 1])],
                ])
                deviatoric_stress = stress - eye * stress.trace() / 3.
                self.mises_stress[ele, igp] = (3./2. * (deviatoric_stress * deviatoric_stress).sum())**0.5


    @ti.kernel
    def get_mises_stress_3d(self, ):
        eye = ti.Matrix([[1., 0., 0.], 
                         [0., 1., 0.], 
                         [0., 0., 1.]])
        for ele in self.elements:
            for igp in range(self.ELE.gaussPoints.shape[0]):
                stress = self.cauchy_stress[ele, igp]
                deviatoric_stress = stress - eye * stress.trace() / 3.
                self.mises_stress[ele, igp] = (3./2. * (deviatoric_stress * deviatoric_stress).sum())**0.5


    def impose_boundary_condition(self, boundary_conditions: dict):
        ### =========== apply the boundary condition ===========
        neumannBCs = boundary_conditions["neumannBCs"]
        dirichletBCs = boundary_conditions["dirichletBCs"]
        
        ### first, apply Neumann BC
        for neumannBC in neumannBCs:
            if "direction" in neumannBC:
                self.neumannBC(neumannBC["face_set"], 
                                load_val=neumannBC["traction"], 
                                load_dir=neumannBC["direction"])
            else:
                self.neumannBC(neumannBC["face_set"], 
                                load_val=neumannBC["traction"])
        
        ### then, apply Dirichlet BC
        if self.geometric_nonlinear == False:
            for dirichletBC in dirichletBCs:
                self.dirichletBC_linearEquations(
                        dirichletBC["node_set"], dirichletBC["dof"], 
                        dirichletBC["val"])
        else:  # large deformation (geometric nonlinear), dirichlet BC is imposed latter at Newton's method
            for dirichletBC in dirichletBCs:
                self.dirichletBC_dof(
                        dirichletBC["node_set"], dirichletBC["dof"], 
                        dirichletBC["val"])


    @ti.kernel 
    def get_deformation_gradient(self, ):
        """get the deformation gradient of each integration point"""
        dm, elements, nodes, gaussPoints = ti.static(
            self.dm, self.elements, self.nodes, self.ELE.gaussPoints)
        eye = ti.Matrix([[0. for _ in range(dm)] for _ in range(dm)])
        for i in range(eye.n):
            eye[i, i] = 1.
        for ele in elements:
            local_us = ti.Matrix([[0. for _ in range(dm)] for _ in range(elements[0].n)], ti.f64)
            local_nodes = ti.Matrix([[0. for _ in range(dm)] for _ in range(elements[0].n)], ti.f64)
            for node_ in range(elements[0].n):
                node = elements[ele][node_]
                for j in ti.static(range(dm)):
                    local_us[node_, j] = self.dof[node * dm + j]  # e.g., displacement
                    local_nodes[node_, j] = nodes[node][j]
            for igp in range(gaussPoints.shape[0]):
                gp = gaussPoints[igp]
                dsdn = self.ELE.dshape_dnat(gp)  # derivative of shape function with respect to natural coodinates
                dnatdX = (local_nodes.transpose() @ dsdn).inverse()  # deformation gradient refers to initial configuration
                dsdX = dsdn @ dnatdX  # strain = (u_node1, u_node2, u_node3) * dsdx
                dudX = local_us.transpose() @ dsdX
                
                ### get the deformation gradient, F = I + dudX
                self.F[ele, igp] = dudX + eye
    

    @ti.kernel 
    def get_strain_smallDeformation(self, ):
        """get the strain directly according to deformation gradient
           the strain here is just for visulization, not for constitutive"""
        dm, elements, gaussPoints, strain = ti.static(
            self.dm, self.elements, self.ELE.gaussPoints, self.strain)
        eye = ti.Matrix([[0. for _ in range(dm)] for _ in range(dm)])
        for i in range(eye.n):
            eye[i, i] = 1.
        
        for ele in elements:
            for igp in range(gaussPoints.shape[0]):
                F = self.F[ele, igp]
                strain[ele, igp] = (F + F.transpose()) / 2. - eye
        

    @ti.kernel 
    def get_strain_largeDeformation(self, ):
        """get the strain directly according to deformation gradient
           the strain here is just for visulization, not for constitutive
           Green strain for large deformation"""
        dm, elements, gaussPoints, strain = ti.static(
            self.dm, self.elements, self.ELE.gaussPoints, self.strain)
        eye = ti.Matrix([[0. for _ in range(dm)] for _ in range(dm)])
        for i in range(eye.n):
            eye[i, i] = 1.
        
        for ele in elements:
            for igp in range(gaussPoints.shape[0]):
                F = self.F[ele, igp]
                strain[ele, igp] = (F.transpose() @ F - eye) / 2.


    @ti.kernel 
    def constitutive_pk2(self, gaussPoints: ti.template()):
        """constitutive use Green's strain and PK2 stress"""
        elements = ti.static(self.elements)
        eye = ti.Matrix([ 
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        for ele in elements:
            for igp in range(gaussPoints.shape[0]):
                F = self.F[ele, igp]

                ### get the Green Strain, E
                E = (F.transpose() @ F - eye) / 2.

                ### get the PK2 stress, voigt notation has been used here, 
                ### modified later by different C at different gauss point
                pk2_voigt = self.material.C @ ti.Vector([E[0, 0], E[1, 1], E[2, 2],
                                                        2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
                pk2 = ti.Matrix([ 
                    [pk2_voigt[0], pk2_voigt[3], pk2_voigt[4]],
                    [pk2_voigt[3], pk2_voigt[1], pk2_voigt[5]],
                    [pk2_voigt[4], pk2_voigt[5], pk2_voigt[2]]
                ])

                ### get the cauchy stress
                self.cauchy_stress[ele, igp] = F @ pk2 @ F.transpose() / F.determinant()


    @ti.kernel 
    def constitutive_infinitesimal(self, gaussPoints: ti.template()):
        """constitutive use Green's strain and PK2 stress"""
        elements = ti.static(self.elements)
        eye = ti.Matrix([ 
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        for ele in elements:
            for igp in range(gaussPoints.shape[0]):
                F = self.F[ele, igp]

                ### get the infinitesimal strain, E
                E = (F + F.transpose()) / 2. - eye

                ### get the PK2 stress, voigt notation has been used here, 
                ### modified later by different C at different gauss point
                s_voigt = self.material.C @ ti.Vector([E[0, 0], E[1, 1], E[2, 2],
                                                    2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
                self.cauchy_stress[ele, igp] = ti.Matrix([ 
                    [s_voigt[0], s_voigt[3], s_voigt[4]],
                    [s_voigt[3], s_voigt[1], s_voigt[5]],
                    [s_voigt[4], s_voigt[5], s_voigt[2]]
                ])


    @ti.kernel
    def constitutive_planeStrain_nonlinear(self, gaussPoints: ti.template()):
        """geometric nonlinear constitutive of plane strain,
           get the stress of each integration point 
           according to deformation gradient"""
        elements, ddsdde = ti.static(self.elements, self.ddsdde)
        eye = ti.Matrix([ 
            [1., 0.],
            [0., 1.],
        ])
        for ele in elements:
            for igp in range(gaussPoints.shape[0]):
                F = self.F[ele, igp]
                
                ### get the Green Strain, E
                E = (F.transpose() @ F - eye) / 2.

                ### get the PK2 stress
                pk2_voigt = ddsdde[ele, igp] @ ti.Vector([E[0, 0], E[1, 1], 
                                                          E[0, 1] + E[1, 0]])
                pk2 = ti.Matrix([ 
                    [pk2_voigt[0], pk2_voigt[2]],
                    [pk2_voigt[2], pk2_voigt[1]]
                ])
                ### get the Cauchy stress  
                self.cauchy_stress[ele, igp] = F @ pk2 @ F.transpose() / F.determinant()
    

    @ti.kernel
    def constitutive_planeStress_nonlinear(self, gaussPoints: ti.template()):
        """geometric nonlinear constitutive of plane stress
           get the stress of each integration point
           constitutive model of plane stress can 
           refer to https://www.comsol.com/blogs/what-is-the-difference-between-plane-stress-and-plane-strain """
        elements, nu = ti.static(self.elements, self.material.poisson_ratio)
        eye_3d = ti.Matrix([ 
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        for ele in elements:
            for igp in range(gaussPoints.shape[0]):
                F = self.F[ele, igp]

                ### get the deformation gradient at 3d
                F_3d = ti.Matrix([[0. for _ in range(3)] for _ in range(3)])
                F_3d[0:2, 0:2] = F[0:2, 0:2]
                F_3d[2, 2] = -nu / (1. - nu) * (F[0, 0] + F[1, 1] - 2.) + 1.  # deformation at z coordinate

                ### get the Green Strain, E
                E = (F_3d.transpose() @ F_3d - eye_3d) / 2.

                ### get the PK2 stress, voigt notation has been used here, 
                ### modified later by different C at different gauss point
                pk2_voigt = self.material.C_6x6 @ ti.Vector([E[0, 0], E[1, 1], E[2, 2],
                                                            2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
                pk2 = ti.Matrix([ 
                    [pk2_voigt[0], pk2_voigt[3], pk2_voigt[4]],
                    [pk2_voigt[3], pk2_voigt[1], pk2_voigt[5]],
                    [pk2_voigt[4], pk2_voigt[5], pk2_voigt[2]]
                ])

                ### get the cauchy stress
                stress = F_3d @ pk2 @ F_3d.transpose() / F_3d.determinant()
                self.cauchy_stress[ele, igp][0:2, 0:2] = stress[0:2, 0:2]
    

    @ti.kernel
    def constitutive_planeStrain_linear(self, gaussPoints: ti.template()):
        """linear constitutive of plane strain,
           get the stress of each integration point 
           according to deformation gradient"""
        elements, ddsdde = ti.static(self.elements, self.ddsdde)
        eye = ti.Matrix([ 
            [1., 0.],
            [0., 1.],
        ])
        for ele in elements:
            for igp in range(gaussPoints.shape[0]):
                F = self.F[ele, igp]
                
                ### get the infinitesimal strain
                E = (F + F.transpose()) / 2. - eye

                ### get the stress
                E_voigt = ti.Vector([E[0, 0], E[1, 1], 
                                     E[0, 1] + E[1, 0]])
                stress_voigt = ddsdde[ele, igp] @ E_voigt
                ### get the Cauchy stress
                self.cauchy_stress[ele, igp] = ti.Matrix([[stress_voigt[0], stress_voigt[2]], 
                                                          [stress_voigt[2], stress_voigt[1]]])


    @ti.kernel
    def constitutive_planeStress_linear(self, gaussPoints: ti.template()):
        """linear constitutive of plane stress
           get the stress of each integration point
           constitutive model of plane stress can 
           refer to https://www.comsol.com/blogs/what-is-the-difference-between-plane-stress-and-plane-strain """
        elements, nu = ti.static(self.elements, self.material.poisson_ratio)
        eye_3d = ti.Matrix([ 
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        for ele in elements:
            for igp in range(gaussPoints.shape[0]):
                F = self.F[ele, igp]

                ### get the deformation gradient at 3d
                F_3d = ti.Matrix([[0. for _ in range(3)] for _ in range(3)])
                F_3d[0:2, 0:2] = F[0:2, 0:2]
                F_3d[2, 2] = -nu / (1. - nu) * (F[0, 0] + F[1, 1] - 2.) + 1.  # deformation at z coordinate

                ### get the infinitesimal strain E
                E = (F_3d + F_3d.transpose()) / 2. - eye_3d

                ### get the stress, voigt notation has been used here, 
                ### modified later by different C at different gauss point
                stress_voigt = self.material.C_6x6 @ ti.Vector([E[0, 0], E[1, 1], E[2, 2],
                                                                2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
                stress = ti.Matrix([ 
                    [stress_voigt[0], stress_voigt[3], stress_voigt[4]],
                    [stress_voigt[3], stress_voigt[1], stress_voigt[5]],
                    [stress_voigt[4], stress_voigt[5], stress_voigt[2]]
                ])
                ### get the cauchy stress
                self.cauchy_stress[ele, igp][0:2, 0:2] = stress[0:2, 0:2]


    def assemble_nodal_force_GN(self, ):
        """assemble the nodal force for GN (geometric nonlinear)"""
        ### get all stresses at integration points by constitutive, modified latter by automatically change consititutive
        if isinstance(self.material, Linear_isotropic_planeStrain):
            self.get_deformation_gradient()
            self.constitutive_planeStrain_nonlinear(self.ELE.gaussPoints)
        elif isinstance(self.material, Linear_isotropic_planeStress):
            self.get_deformation_gradient()
            self.constitutive_planeStress_nonlinear(self.ELE.gaussPoints)
        elif isinstance(self.material, Linear_isotropic):
            self.get_deformation_gradient()
            self.constitutive_pk2(self.ELE.gaussPoints)
        else:
            print("\033[31;1m error! currently we only support these types of materials: "
                  "plane strain, plane stress and 3D linear elastic \033[0m")
        ### get dsdx and vol
        self.get_dsdx_and_vol()
        ### assemble to nodal force
        self.assemble_nodal_force_GN_kernel()


    @ti.kernel 
    def get_dsdx_and_vol(self, ):
        dm, elements, nodes, dof, gaussPoints  = ti.static(
            self.dm, self.elements, self.nodes, self.dof, self.ELE.gaussPoints)
        for ele in self.elements:
            localNodes = ti.Matrix([[0. for i in range(dm)] for j in range(elements[0].n)], ti.f64)
            for i in range(localNodes.n):
                for j in range(localNodes.m):
                    localNodes[i, j] = nodes[elements[ele][i]][j] + \
                                        dof[elements[ele][i] * dm + j]  # nodes coos at current configuration
            
            ### get dsdx and vol of each integration point
            for igp in range(gaussPoints.shape[0]):
                gp = gaussPoints[igp]
                dsdn = self.ELE.dshape_dnat(gp)  # derivative of shape function with respect to natural coodinates
                dnatdx = (localNodes.transpose() @ dsdn).inverse()  # at current configuration
                self.dsdx[ele, igp] = dsdn @ dnatdx  # dshape / dx, gradient at current configuration
                self.vol[ele, igp] = (localNodes.transpose() @ dsdn).determinant() * self.ELE.gaussWeights[igp]

    
    @ti.kernel
    def assemble_nodal_force_GN_kernel(self, ):
        dm, nodal_force, nodeEles, nodes, gaussPoints, dsdx, cauchy_stress = ti.static(
            self.dm, self.nodal_force, self.nodeEles, self.nodes, 
            self.ELE.gaussPoints, self.dsdx, self.cauchy_stress)
        ### refresh the nodal force before assembling
        for i in nodal_force:
            nodal_force[i] = 0.
        ### begin to assemble
        for node0 in nodes:
            for iele in range(nodeEles[0].n):
                if nodeEles[node0][iele] != -1:
                    ele = nodeEles[node0][iele]

                    ### get the sequence of this node in the element
                    nid = get_index_ti(self.elements[ele], node0)
                    if nid == -1:
                        print("\033[31;1m Error, index not found. nid = -1 \033[0m")

                    ### assemble stress to the nodal force
                    for igp in range(gaussPoints.shape[0]):
                        dsdx_x_stress = dsdx[ele, igp][nid, :] @ cauchy_stress[ele, igp]
                        for i in range(dm):
                            nodal_force[node0 * dm + i] = \
                            nodal_force[node0 * dm + i] + dsdx_x_stress[i] * self.vol[ele, igp]


    def solve(self, inp: Inp_info, show_newton_steps: bool=False, save2path: str=None):
        """solved by multiple time increments, each increment calls slove_inc()"""
        max_inc = inp.time_incs["max_inc"]
        min_inc = inp.time_incs["min_inc"]
        max_time = inp.time_incs["max_time"]
        self.dt = inp.time_incs["ini_inc"]
        
        neumannBCs = copy.deepcopy(inp.neumann_bc_info)
        dirichletBCs = copy.deepcopy(inp.dirichlet_bc_info)
        for dirichletBC in dirichletBCs:
            node_set = ti.field(ti.i32, shape=(len(dirichletBC["node_set"])))
            node_set.from_numpy(np.array([*dirichletBC["node_set"]]))
            dirichletBC["node_set"] = node_set  # replace the original node set by field
        boundary_conditions = {"neumannBCs": neumannBCs, "dirichletBCs": dirichletBCs}

        ### whether show the body during time step and Newton's step
        if show_newton_steps and self.geometric_nonlinear:
            windowLength = 512
            if not isinstance(self.ELE, Element_linear_triangular):
                window = ti.ui.Window('show body', (windowLength, windowLength))
            else:
                window = ti.GUI('show body', res=(windowLength, windowLength))
        else: window = None

        ### now start the time increments
        kinc = -1  # the number of time increment
        while self.time1 < max_time:
            kinc += 1
            self.time1 = min(self.time0 + self.dt, max_time)
            print("\033[40;33;1m >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                                ">>>>> kinc = {}, time0 = {}, dt = {} \033[0m".format(kinc, self.time0, self.dt))
            load_ratio = self.time1 / max_time
            ### set the boundary according to load ratio
            for id, neumannBC in enumerate(neumannBCs):
                neumannBC["traction"] = inp.neumann_bc_info[id]["traction"] * load_ratio
            for id, dirichletBC in enumerate(dirichletBCs):
                dirichletBC["val"] = inp.dirichlet_bc_info[id]["val"] * load_ratio
            ### advance a time increment
            converged, newton_loop = self.advance_inc(inp, boundary_conditions, 
                                                      show_newton_steps, save2path, window)  # update self.dof
            if not converged:
                self.time1 = self.time0
                self.dt /= 4.
                a_from_b(self.dof, self.dof_old)
                kinc -= 1
                if self.dt < min_inc:
                    print("\033[31;1m allowable minimum dt is reached, "
                          "Newton's method not converges, solution is not found. \033[0m")
                    break
                continue
            ### increast dt if fast convergence occurs in previous dt
            if newton_loop <= 8:
                self.dt = min(self.dt * 1.5, max_inc)
            a_from_b(self.dof_old, self.dof)  # self.dof_old[:] = self.dof[:]
            self.time0 = self.time1
    
    
    def advance_inc(self, inp: Inp_info, boundary_conditions: dict, 
                    show_newton_steps: bool=False, save2path: str=None, 
                    window: Union[ti.ui.Window, ti.GUI]=None, 
                  ) -> Tuple[bool, int]:

        def inside_relaxation():
            self.assemble_nodal_force_GN(); self.assemble_sparseMtrx()  # use new dofs to compute nodal force
            c_equals_a_minus_b(self.residual_nodal_force, self.nodal_force, self.rhs)
            self.dirichletBC_forNewtonMethod(boundary_conditions["dirichletBCs"])
            residual = field_norm(self.residual_nodal_force)
            print("\033[32;1m residual = {} \033[0m".format(residual))
            if show_newton_steps:
                self.show_window(window, save2path, newton_loop, relax_loop)
            return residual
        

        """solve at each time increment"""
        geometric_nonlinear = inp.geometric_nonlinear
        print("\033[35;1m >>> geometric nonlinear is {}. \033[0m".format(
            {False: "off", True: "on"}[geometric_nonlinear]))

        print("\033[32;1m now we begin to assemble the sparse matrix \033[0m"); time0 = time.time()
        self.get_dsdx_and_vol()
        self.assemble_sparseMtrx()
        print("\033[32;1m sparse matrix assembling is finished  \033[0m"); time1 = time.time()
        if not self.compiled:
            self.compiled = True
            print("\033[35;1m assemble_sparseMtrx's compiling time is {} s\033[0m".format(time1 - time0))
        else:
            print("time for assemble is {} s".format(time1 - time0))

        ### impost boundary condition at the initial state
        self.impose_boundary_condition(boundary_conditions)
        
        if geometric_nonlinear == False:  # small deformation
            self.solve_dof()
            return True, 0
        
        else:  # large deformation, use newton method
            
            ### compute nodal force for large deformation
            self.assemble_nodal_force_GN(); self.assemble_sparseMtrx()
            c_equals_a_minus_b(self.residual_nodal_force, self.nodal_force, self.rhs)
            self.dirichletBC_forNewtonMethod(boundary_conditions["dirichletBCs"])
            pre_residual = field_norm(self.residual_nodal_force)
            if not hasattr(self, "ini_residual"):
                self.ini_residual = pre_residual
            print("\033[40;33;1m initial residual_nodal_force = {} \033[0m".format(self.ini_residual))
            if show_newton_steps:
                self.show_window(window, save2path, newton_loop=0, relax_loop=0)

            if self.ini_residual < 1.e-9:
                print("\033[32;1m good! nonlinear converge! \033[0m")
            else:
                newton_loop = -1
                while pre_residual / (self.ini_residual + 1.e-30) >= 0.01:  # not convergent
                    
                    newton_loop += 1
                    if newton_loop >= 24:
                        return False, newton_loop  # Newton's method has not converged

                    solver = self.solve_dof()  # dofs = dofs - K^(-1) * residual

                    self.assemble_nodal_force_GN(); self.assemble_sparseMtrx()  # use new dofs to compute nodal force
                    ### self.residual_nodal_force = self.nodal_force - self.rhs
                    c_equals_a_minus_b(self.residual_nodal_force, self.nodal_force, self.rhs)
                    self.dirichletBC_forNewtonMethod(boundary_conditions["dirichletBCs"])
                    residual = field_norm(self.residual_nodal_force)
                    print("\033[40;33;1m newton_loop = {}, residual_nodal_force = {} \033[0m".format(newton_loop, residual))
                    if show_newton_steps:
                        self.show_window(window, save2path, newton_loop, relax_loop=0)

                    ### boost Newton's method by going a larger step if residual force is declining
                    relax_loop = -1; relaxation = 1.
                    relaxation = 1.  #  further_step_ratio
                    while 0.1 * pre_residual < residual < pre_residual:  # when residual declines, go on this direction further
                        new_residual = residual
                        relax_loop += 1
                        if relax_loop >= 10:
                            break
                        print("\033[35;1m further_step_ratio = {} \033[0m".format(relaxation))
                        ### self.dof -= relaxation * solver.x
                        a_equals_b_plus_c_mul_d(self.dof, self.dof, -relaxation, solver.x)
                        residual = inside_relaxation()
                        if residual > new_residual:
                            a_equals_b_plus_c_mul_d(self.dof, self.dof, +relaxation, solver.x)
                            residual = inside_relaxation()
                            relaxation *= 0.5
                    
                    ### relaxation for Newton's method when residual gets bigger
                    relax_loop = -1; relaxation = 1.
                    while residual > pre_residual:
                        relax_loop += 1
                        if relax_loop >= 2:
                            break
                        relaxation *= 0.5
                        print("\033[35;1m relaxation = {} \033[0m".format(relaxation))
                        ### self.dof += (1. - relaxation) * solver.x, i.e., recover dof, then update with relaxation  
                        a_equals_b_plus_c_mul_d(self.dof, self.dof, (1. - relaxation), solver.x)
                        field_multiply(solver.x, relaxation)
                        residual = inside_relaxation()

                    pre_residual = residual
            return True, newton_loop  # Newton's method converges


    def show_window(self, window, save2path: str=None, newton_loop: int=0, relax_loop: int=0):
        self.compute_strain_stress()
        if not isinstance(self.ELE, Element_linear_triangular):
            self.ELE.extrapolate(self.mises_stress, self.nodal_vals)
            self.body.show(window, self.dof, self.nodal_vals)
        else: 
            self.body.show2d(window, disp=self.dof, 
                            field=self.mises_stress.to_numpy(dtype=np.float64), 
                            save2path="{}_{}_{}_{}.png".format(save2path, self.time1, newton_loop, relax_loop) if save2path else None)


if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)
    fileName = input("\033[32;1m please give the .inp format's "
                        "input file path and name: \033[0m")
    ### for example, fileName = ./tests/elliptic_membrane/element_linear/ellip_membrane_linEle_localVeryFine.inp
    inp = Inp_info(fileName)
    nodes, eSets = inp.nodes, inp.eSets
    body = Body(nodes=nodes, elements=list(eSets.values())[0])
    ele_type = list(eSets.keys())[0]
    if ele_type[0:3] == "CPS":
        material = Linear_isotropic_planeStress(modulus=inp.materials["Elastic"][0], 
                                                poisson_ratio=inp.materials["Elastic"][1])
    elif ele_type[0:3] == "CPE":
        material = Linear_isotropic_planeStrain(modulus=inp.materials["Elastic"][0], 
                                                poisson_ratio=inp.materials["Elastic"][1])

    equationSystem = System_of_equations(body, material, inp.geometric_nonlinear)

    equationSystem.solve(inp)
    print("\033[40;33;1m equationSystem.dof = \n{} \033[0m".format(equationSystem.dof.to_numpy()))

    ### show the body
    equationSystem.compute_strain_stress()
    stress = equationSystem.mises_stress.to_numpy()
    print("\033[35;1m maximum mises stress = {} MPa \033[0m".format(abs(stress).max()))
    print("\033[40;33;1m max dof (disp) = {} \033[0m".format(field_abs_max(equationSystem.dof)))
    
    windowLength = 512
    gui = ti.GUI('show body', res=(windowLength, windowLength))
    while gui.running:
        equationSystem.body.show2d(gui, disp=equationSystem.dof, field=stress)