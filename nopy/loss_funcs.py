import numpy as np
from functools import partial


def make_func(name, f_scale=1.0):
    if name == 'linear':
        return partial(linear, f_scale=f_scale)
    elif name == 'soft_l1':
        return partial(soft_l1, f_scale=f_scale)
    elif name == 'huber':
        return partial(huber, f_scale=f_scale)
    elif name == 'cauchy':
        return partial(cauchy, f_scale=f_scale)
    elif name == 'arctan':
        return partial(arctan, f_scale=f_scale)
    else:
        raise ValueError('Unsupported loss function.')


def linear(z: np.ndarray, f_scale):
    """
    rho(z) = z
    No loss function at all.
    :param z: z = f(x)**2
    :param f_scale: no effect with linear
    :return: 
    """
    loss = np.empty((3, z.shape[0]), dtype=np.float64)
    loss[0, :] = z
    loss[1, :] = 1
    loss[2, :] = 0
    return loss


def soft_l1(z: np.ndarray, f_scale):
    """
    rho(z) = 2 * ((1 + z)**0.5 - 1)
    The smooth approximation of l1 (absolute value) loss. Usually a good choice for robust least squares.
    :param z: z = f(x)**2
    :param f_scale: rho_(f**2) = C**2 * rho(f**2 / C**2), where C is f_scale
    :return: 
    """
    loss = np.empty((3, z.shape[0]), dtype=np.float64)

    c2 = f_scale * f_scale
    ic2 = 1.0 / c2
    z = ic2 * z
    sqrt_1pz = np.sqrt(z + 1)

    loss[0, :] = c2 * 2 * (sqrt_1pz - 1)
    loss[1, :] = 1 / sqrt_1pz
    loss[2, :] = -ic2 * 0.5 * np.power(loss[1, :], 3)
    return loss


def huber(z: np.ndarray, f_scale):
    """
    rho(z) = z if z <= 1 else 2*z**0.5 - 1
    Works similarly to 'soft_l1'.
    :param z: z = f(x)**2
    :param f_scale: rho_(f**2) = C**2 * rho(f**2 / C**2), where C is f_scale
    :return: 
    """
    loss = np.empty((3, z.shape[0]), dtype=np.float64)

    c2 = f_scale * f_scale
    ic2 = 1.0 / c2
    z = ic2 * z
    cond1 = z <= 1
    cond2 = z > 1

    loss[0, :] = c2 * np.piecewise(z, [cond1, cond2], [lambda x: x, lambda x: 2*np.sqrt(x) - 1])
    loss[1, :] = np.piecewise(z, [cond1, cond2], [1, lambda x: 1/np.sqrt(x)])
    loss[2, :] = ic2 * np.piecewise(z, [cond1, cond2], [0, lambda x: -0.5 * np.power(x, -1.5)])
    return loss


def cauchy(z: np.ndarray, f_scale):
    """
    rho(z) = ln(1 + z)
    Severely weakens outliers influence, but may cause difficulties in optimization process.
    :param z: z = f(x)**2
    :param f_scale: rho_(f**2) = C**2 * rho(f**2 / C**2), where C is f_scale
    :return: 
    """
    loss = np.empty((3, z.shape[0]), dtype=np.float64)

    c2 = f_scale * f_scale
    ic2 = 1.0 / c2
    z = ic2 * z
    zp1 = z + 1

    loss[0, :] = c2 * np.log(zp1)
    loss[1, :] = 1 / zp1
    loss[2, :] = -ic2 * (loss[1, :] ** 2)
    return loss


def arctan(z: np.ndarray, f_scale):
    """
    rho(z) = arctan(z)
    Limits a maximum loss on a single residual, has properties similar to ‘cauchy’.
    :param z: z = f(x)**2
    :param f_scale: rho_(f**2) = C**2 * rho(f**2 / C**2), where C is f_scale
    :return: 
    """
    loss = np.empty((3, z.shape[0]), dtype=np.float64)

    c2 = f_scale * f_scale
    ic2 = 1.0 / c2
    z = ic2 * z

    loss[0, :] = c2 * np.arctan(z)
    loss[1, :] = 1.0 / (1.0 + z ** 2)
    loss[2, :] = -ic2 * 2 * z * (loss[1, :] ** 2)
    return loss
