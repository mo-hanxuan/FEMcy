import taichi as ti
from time import time

from readInp import *
from material import *
import yaml


@ti.data_oriented
class ConjugateGradientSolver_rowMajor:

    def __init__(self, spm: ti.template(), sparseIJ: ti.template(),  # this is used to define the sparse matrix
                 b: ti.template(),
                 eps=1.e-3,  # the allowable relative error of residual 
                 ):
        self.A = spm  # sparse matrix, also called stiffness matrix or coefficient matrix
        self.ij = sparseIJ  # for each row, record the colume index of sparse matrix
        self.b = b  # the right hand side (rhs) of the linear system

        self.x = ti.field(ti.f64, b.shape[0]); self.x.fill(0.)  # the solution x
        self.r = ti.field(ti.f64, b.shape[0])  # the residual
        self.d = ti.field(ti.f64, b.shape[0])  # the direction of change of x
        self.M = ti.field(ti.f64, b.shape[0]); self.M_init()  # the inverse of precondition diagonal matrix, M^(-1) actually

        self.Ad = ti.field(ti.f64, b.shape[0])  # A multiple d
        self.eps = eps
    

    @ti.func
    def A_get(self, i, j):
        target_j = 0
        for j0 in range(self.ij[i][0]):
            if self.ij[i][j0 + 1] == j:
                target_j = j0
        return self.A[i][target_j]


    @ti.kernel
    def M_init(self, ):  # initialize the precondition diagonal matrix
        for i in self.M:
            self.M[i] = 1. / self.A_get(i, i)
    

    @ti.kernel
    def compute_Ad(self, ):  # compute A multiple d
        for i in self.A:
            self.Ad[i] = 0.
            for j0 in range(self.ij[i][0]):
                self.Ad[i] = self.Ad[i] + self.A[i][j0] * self.d[self.ij[i][j0 + 1]]
    

    @ti.kernel
    def r_d_init(self, ):  # initial residual r and direction d
        for i in self.r:  # r0 = b - Ax0 = b
            self.r[i] = self.b[i]
        for i in self.d:
            self.d[i] = self.M[i] * self.r[i]  # d0 = M^(-1) * r
    

    @ti.kernel
    def rmax(self, ) -> float:  # max of abs(r), modified latter by reduce_max
        rm = 0.
        for i in self.r:
            ti.atomic_max(rm, ti.abs(self.r[i]))
        return rm
    

    @ti.kernel
    def compute_rMr(self, ) -> float:  # r * M^(-1) * r
        rMr = 0.
        for i in self.r:
            rMr += self.r[i] * self.M[i] * self.r[i]
        return rMr
    

    @ti.kernel 
    def update_x(self, alpha: float):
        for j in self.x:
            self.x[j] = self.x[j] + alpha * self.d[j]
    

    @ti.kernel
    def update_r(self, alpha: float):
        for j in self.r:
            self.r[j] = self.r[j] - alpha * self.Ad[j]
    

    @ti.kernel 
    def update_d(self, beta: float):
        for j in self.d:
            self.d[j] = self.M[j] * self.r[j] + beta * self.d[j]
    

    @ti.kernel
    def dot_product(self, y: ti.template(), z: ti.template()) -> float:
        res = 0.
        for i in y:
            res += y[i] * z[i]
        return res


    def solve(self, ):
        self.r_d_init()
        r0 = self.rmax()  # the inital residual scale
        print("\033[32;1m the initial residual scale is {} \033[0m".format(r0))
        time_outloop = time()

        for i in range(self.b.shape[0]):  # CG will converge within at most b.shape[0] loops
            t0 = time()
            self.compute_Ad()
            rMr = self.compute_rMr()
            alpha = rMr / self.dot_product(self.d, self.Ad)
            self.update_x(alpha)
            self.update_r(alpha)
            beta = self.compute_rMr() / rMr
            self.update_d(beta)
            
            rmax = self.rmax()  # the infinite norm of residual, shold be modified latter to the reduce max
            t1 = time()

            if i % 32 == 0:
                print("\033[35;1m the {}-th loop, norm of residual is {}, in-loop time is {} s\033[0m".format(
                    i, rmax, t1 - t0
                ))
            if rmax < self.eps * r0:  # converge?
                print("\033[35;1m the {}-th loop, norm of residual is {}, in-loop time is {} s\033[0m".format(
                    i, rmax, t1 - t0
                ))
                break
        print("\033[32;1m CG solver's computation time is {} s\033[0m".format(time() - time_outloop))


if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)
    dataFile = "./tests/example_linearSystem.yml"
    data = yaml.load(open(dataFile, "r").read())
    sparseMtrx_I = data["sparseMtrx_I"]
    sparseMtrx_J = data["sparseMtrx_J"]
    sparseMtrx_val = data["sparseMtrx_val"]
    rhs_ = data["rhs"]
    print("\033[35;1m rhs = \n{} \033[0m".format(rhs_))

    result = ti.field(ti.f64, shape=(len(rhs_), ))
    result.from_numpy(np.array(data["result"]))

    Is = ti.field(ti.i32, shape=(len(sparseMtrx_val), ))
    Is.from_numpy(np.array(sparseMtrx_I))
    Js = ti.field(ti.i32, shape=(len(sparseMtrx_val), ))
    Js.from_numpy(np.array(sparseMtrx_J))
    sparse_vals = ti.field(ti.f64, shape=(len(sparseMtrx_val), ))
    sparse_vals.from_numpy(np.array(sparseMtrx_val))

    ### row major sparse matrix
    maxI = max(sparseMtrx_I) + 1  # len of I actually
    print("\033[31;1m type(maxI) = {} \033[0m".format(type(maxI)))
    rowLens = np.zeros(maxI, dtype=np.int64)
    for i in sparseMtrx_I:
        rowLens[i] += 1
    max_row_lens = rowLens.max()
    spm_ = [[] for _ in range(maxI)]
    sparseIJ_ = [[] for _ in range(maxI)]
    for idx in range(len(sparseMtrx_val)):
        i, j = sparseMtrx_I[idx], sparseMtrx_J[idx]
        spm_[i].append(sparseMtrx_val[idx])
        sparseIJ_[i].append(j)
    spm = np.zeros(shape=(maxI, max_row_lens), dtype=np.float64)
    sparseIJ = -np.ones(shape=(maxI, max_row_lens), dtype=np.int64)
    for i in range(len(spm)):
        spm[i, :len(spm_[i])] = spm_[i][:]
        sparseIJ[i, :len(sparseIJ_[i])] = sparseIJ_[i][:]
    
    ### transform to taichi field
    A = ti.Vector.field(spm.shape[1], ti.f64, shape=(spm.shape[0], ))
    A.from_numpy(spm)
    ij = ti.Vector.field(sparseIJ.shape[1], ti.i32, shape=(sparseIJ.shape[0], ))
    ij.from_numpy(sparseIJ)

    ### sparse spatial data structure in taichi
    rhs = ti.field(ti.f64, shape=(len(rhs_), ))
    rhs.from_numpy(np.array(rhs_))

    print("rhs = {}, rhs.shape = {}".format(rhs, rhs.shape))

    t0 = time()
    CGSolver = ConjugateGradientSolver_rowMajor(spm=A, sparseIJ=ij, b=rhs)
    CGSolver.solve()
    print("total time for CG class builder and solver is {} s".format(time() - t0))
    print("\033[32;1m CGSolver.x = \n{} \033[0m".format(CGSolver.x))