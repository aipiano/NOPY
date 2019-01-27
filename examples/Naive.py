import nopy
import numpy


def f1(x1):
    return x1 - 1


def f2(x2):
    return x2 - 3


x1 = numpy.array([-1], dtype=numpy.float64)
x2 = numpy.array([0], dtype=numpy.float64)


problem = nopy.LeastSquaresProblem()
problem.add_residual_block(1, f1, x1, jac_func='2-point')
problem.add_residual_block(1, f2, x2, jac_func='2-point')

problem.solve()

print(x1)
print(x2)
