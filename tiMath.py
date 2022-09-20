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
def a_from_b(a: ti.template(), b: ti.template()):
    """a = b"""
    for I in ti.grouped(a):
        a[I] = b[I]


@ti.kernel
def field_abs_max(f: ti.template()) -> float:
    """get the maximum absolute value of a scaler field"""
    ans = 0.
    for I in ti.grouped(f):
        ti.atomic_max(ans, ti.abs(f[I]))
    return ans


@ti.kernel 
def field_max(f: ti.template()) -> float:
    """get the max value of a scaler field"""
    ans = -float("inf")
    for I in ti.grouped(f):
        ti.atomic_max(ans, f[I])
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


if __name__ == "__main__":

    while True:
        a, b = map(int, input("a, b = ").split(","))
        print("x, y =", fraction_reduction(a, b))