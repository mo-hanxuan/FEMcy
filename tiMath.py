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


def fraction_reduction(a: int, b: int):
    """fraction a/b is reduced by x/y
       https://blog.csdn.net/Cosmos53/article/details/116330862 """
    x, y = a, b
    while b > 0:
        a, b = b, a % b
    return x // a, y // a


if __name__ == "__main__":

    while True:
        a, b = map(int, input("a, b = ").split(","))
        print("x, y =", fraction_reduction(a, b))