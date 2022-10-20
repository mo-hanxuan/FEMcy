"""some functions for mathmatical usages"""
import taichi as ti


@ti.kernel
def c_equals_a_minus_b(c: ti.template(), a: ti.template(), b: ti.template()):
    """c = a - b"""
    for I in ti.grouped(c):
        c[I] = a[I] - b[I]


@ti.kernel
def a_equals_b_plus_c_mul_d(a:ti.template(), b:ti.template(), c: float, d: ti.template()):
    """a = b + c * d"""
    for I in ti.grouped(a):
        a[I] = b[I] + c * d[I]


@ti.kernel
def field_abs_max(f: ti.template()) -> float:
    """get the maximum absolute value of a scaler field"""
    ans = 0.
    for I in ti.grouped(f):
        ti.atomic_max(ans, ti.abs(f[I]))
    return ans


@ti.kernel 
def field_norm(f: ti.template()) -> float:
    """get modified Euclidean norm of the scaler field, (the modified 2nd norm) """
    ans = 0.
    for I in ti.grouped(f):
        ans += f[I] ** 2
    N = 1
    for i in ti.static(range(len(f.shape))):
        N *= f.shape[i]
    return (ans / N) ** 0.5


@ti.kernel 
def field_max(f: ti.template()) -> float:
    """get the max value of a scaler field"""
    ans = -float("inf")
    for I in ti.grouped(f):
        ti.atomic_max(ans, f[I])
    return ans


@ti.kernel 
def vectorField_max(f: ti.template()) -> float:
    """get the max value of a vector field"""
    ans = -float("inf")
    for I in ti.grouped(f):
        ti.atomic_max(ans, f[I].max())
    return ans


@ti.kernel 
def field_min(f: ti.template()) -> float:
    """get the min value of a scaler field"""
    ans = float("inf")
    for I in ti.grouped(f):
        ti.atomic_min(ans, f[I])
    return ans


@ti.kernel
def field_multiply(field: ti.template(), num: float):
    for i in field:
        field[i] *= num


@ti.kernel
def field_add(field: ti.template(), num: float):
    for i in field:
        field[i] = field[i] + num  # do not use +=, beacuse that is atomic add which could lose precision

@ti.kernel
def field_addVec(field: ti.template(), vec: ti.template()):
    for i in field:
        field[i] = field[i] + vec  # do not use +=, beacuse that is atomic add which could lose precision


@ti.func
def sorted_tiVec(arr):
    """for small vector, naive bubble sort is acceptable"""
    for i in ti.static(range(1, arr.n)):
        for j in ti.static(range(0, arr.n - i)):
            if arr[j] > arr[j+1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


@ti.func
def get_index_ti(arr, val) -> int:
    """get the index of val in vector arr"""
    index = -1
    for i in ti.static(range(arr.n)):
        if arr[i] == val:
            index = i
    return index


@ti.kernel 
def scalerField_from_matrixField(f1: ti.template(), f2: ti.template(), i: int, j: int):
    """fill the scaler field with index [i, j] of a matrix field"""
    for I in ti.grouped(f1):
        f1[I] = f2[I][i, j]


def fraction_reduction(a: int, b: int):
    """fraction a/b is reduced by x/y
       https://blog.csdn.net/Cosmos53/article/details/116330862 """
    x, y = a, b
    while b > 0:
        a, b = b, a % b
    return x // a, y // a


def relative_error(a, b):
    """
    get the relative error between a and b
    """
    maxVal = max(abs(a), abs(b))
    if maxVal > 1.e-9:
        return abs(a - b) / maxVal  # return relative error
    else:
        return abs(a - b)  # return absolute error if a and b are almost 0


@ti.func
def vec_mul_voigtMtrx(vec, mtrx):  # vector (line of a matrix) multiply voigt matrix
    """
    dot product of vector and voigt matrix
    for 2-dimension
        matrix = [
            [mtrx[0], mtrx[2]],
            [mtrx[2], mtrx[1]]
        ]
        thus, vec * matrix = [
            vec[0] * mtrx[0, :] + vec[1] * mtrx[2, :], 
            vec[0] * mtrx[2, :] + vec[1] * mtrx[1, :], 
        ]
    for 3-dimension
        matrix = [
            [mtrx[0], mtrx[3], mtrx[4]],
            [mtrx[3], mtrx[1], mtrx[5]],
            [mtrx[4], mtrx[5], mtrx[2]],
        ]
        thus, vec * matrix = [
            vec[0] * mtrx[0, :] + vec[1] * mtrx[3, :] + vec[2] * mtrx[4, :], 
            vec[0] * mtrx[3, :] + vec[1] * mtrx[1, :] + vec[2] * mtrx[5, :], 
            vec[0] * mtrx[4, :] + vec[1] * mtrx[5, :] + vec[2] * mtrx[2, :], 
        ]
    """
    ans = ti.Matrix.zero(ti.f64, vec.m, mtrx.m) #([[0. for _ in range(mtrx.m)] for _ in range(vec.m)])
    if vec.m == 2:
        ans[0, :] = vec[0] * mtrx[0, :] + vec[1] * mtrx[2, :]
        ans[1, :] = vec[0] * mtrx[2, :] + vec[1] * mtrx[1, :]
    elif vec.m == 3:
        ans[0, :] = vec[0] * mtrx[0, :] + vec[1] * mtrx[3, :] + vec[2] * mtrx[4, :]
        ans[1, :] = vec[0] * mtrx[3, :] + vec[1] * mtrx[1, :] + vec[2] * mtrx[5, :]
        ans[2, :] = vec[0] * mtrx[4, :] + vec[1] * mtrx[5, :] + vec[2] * mtrx[2, :]
    return ans


if __name__ == "__main__":

    while True:
        a, b = map(int, input("a, b = ").split(","))
        print("x, y =", fraction_reduction(a, b))