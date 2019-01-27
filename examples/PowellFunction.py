import nopy
import numpy as np
import math


def f1(x1, x2):
    return x1 + 10*x2


def f2(x3, x4):
    return math.sqrt(5) * (x3 - x4)


def f3(x2, x3):
    y = x2 - 2*x3
    return y[0] * y[0]


def f4(x1, x4):
    y = x1 - x4
    return math.sqrt(10) * y[0] * y[0]


def main():
    x1 = np.array([2], dtype=np.float64)
    x2 = np.array([3], dtype=np.float64)
    x3 = np.array([0], dtype=np.float64)
    x4 = np.array([-7], dtype=np.float64)

    problem = nopy.LeastSquaresProblem()
    problem.add_residual_block(1, f1, x1, x2, jac_func='2-point')
    problem.add_residual_block(1, f2, x3, x4, jac_func='2-point')
    problem.add_residual_block(1, f3, x2, x3, jac_func='2-point')
    problem.add_residual_block(1, f4, x1, x4, jac_func='2-point')

    # problem.fix_variables(x1)
    # problem.unfix_variables(x1)
    problem.solve()

    print(x1)
    print(x2)
    print(x3)
    print(x4)


if __name__ == '__main__':
    main()
