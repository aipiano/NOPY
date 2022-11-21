import numpy as np
import scipy.optimize
import scipy.sparse
import nopy.loss_funcs as loss_funcs
import nopy.auto_diff as auto_diff


class _ResidualBlock:
    """
    DO NOT USE THIS CLASS DIRECTLY!
    """
    def __init__(self):
        self.dim_residual = 0
        self.dim_variable = 0
        self.residual_func = None
        self.loss_func = None
        self.jac_func = None
        self.jac_sparsity = None
        self.row_range = (0, 0)
        self.col_ranges = []


class LeastSquaresProblem:
    def __init__(self):
        self.__dim_variable = 0
        self.__dim_residual = 0
        self.__address_col_range_map = {}
        self.__col_range_variable_map = {}
        self.__residual_blocks = []
        self.__x0 = None
        self.__jac_sparsity = None
        self.__fixed_variable_address = set()
        self.__address_bounds_map = {}
        self.__bounds = None

    def add_residual_block(self, dim_residual, residual_func, *variables, loss_func='linear', f_scale=1.0,
                           jac_func='2-point', jac_sparsity=None):
        """
        add a residual block to the problem.
        :param dim_residual: int
            the dimension m of residual
        :param residual_func: callable
            Function which computes the vector of residuals, with the signature func(*variables).
            It must return a 1-d array_like of shape (m,) or a scalar.
        :param variables: each variable must be a np.array with the shape of (n,).
        :param loss_func: string or callable
            Determines the loss function. The following keyword values are allowed:
            - 'linear' (default) : rho(z) = z. Gives a standard least-squares problem.
            - 'soft_l1' : rho(z) = 2 * ((1 + z)**0.5 - 1). 
              The smooth approximation of l1 (absolute value) loss. Usually a good choice for robust least squares.
            - 'huber' : rho(z) = z if z <= 1 else 2*z**0.5 - 1. 
              Works similarly to ‘soft_l1’.
            - 'cauchy' : rho(z) = ln(1 + z). 
              Severely weakens outliers influence, but may cause difficulties in optimization process.
            - 'arctan' : rho(z) = arctan(z). 
              Limits a maximum loss on a single residual, has properties similar to ‘cauchy’.
            If callable, it must take a 1-d ndarray z=f**2 and return an array_like with shape (3, m) where row 0 
            contains function values, row 1 contains first derivatives and row 2 contains second derivatives. 
            Method ‘lm’ supports only ‘linear’ loss.
        :param f_scale: float
            Value of soft margin between inlier and outlier residuals, default is 1.0. 
            The loss function is evaluated as follows rho_(f**2) = C**2 * rho(f**2 / C**2), where C is f_scale, 
            and rho is determined by loss parameter. This parameter has no effect with loss='linear', 
            but for other loss values it is of crucial importance.
            NOTE! Too small f_scale may course data overflow!
        :param jac_func: '2-point','3-point' or a callable object
            Method of computing the Jacobian matrix
            (an m-by-n matrix, where element (i, j) is the partial derivative of f[i] with respect to x[j],
            where x is the concatenation of all input variables).
            The keywords select a finite difference scheme for numerical estimation.
            The scheme ‘3-point’ is more accurate, but requires twice as many operations as ‘2-point’ (default).
        :param jac_sparsity: {None, array_like, sparse_matrix}
            the sparsity of the jacobian matrix with the same shape of the jacobian matrix.
            A zero entry means that a corresponding element in the Jacobian is identically zero.
        :return:
        """
        residual_block = _ResidualBlock()

        assert dim_residual > 0
        residual_block.dim_residual = dim_residual

        assert callable(residual_func)
        residual_block.residual_func = residual_func

        if callable(loss_func):
            residual_block.loss_func = loss_func
        else:
            residual_block.loss_func = loss_funcs.make_func(loss_func, f_scale=f_scale)

        residual_block.row_range = (self.__dim_residual, self.__dim_residual + dim_residual)
        self.__dim_residual += dim_residual

        residual_block.col_ranges = []
        residual_block.dim_variable = 0
        for variable in variables:
            assert len(variable.shape) == 1
            dim = variable.shape[0]
            residual_block.dim_variable += dim

            address = variable.__array_interface__['data'][0]
            if address not in self.__address_col_range_map:
                new_range = (self.__dim_variable, self.__dim_variable + dim)
                self.__dim_variable += dim
                self.__address_col_range_map[address] = new_range
                self.__col_range_variable_map[new_range] = variable
            residual_block.col_ranges.append(self.__address_col_range_map[address])

        if callable(jac_func):
            residual_block.jac_func = jac_func
        else:
            residual_block.jac_func = auto_diff.make_func(jac_func,
                                                          (residual_block.dim_residual, residual_block.dim_variable),
                                                          residual_block.residual_func)

        if jac_sparsity is None:
            residual_block.jac_sparsity = None
        else:
            assert jac_sparsity.shape[0] == dim_residual and jac_sparsity.shape[1] == residual_block.dim_variable
            residual_block.jac_sparsity = jac_sparsity.copy()

        self.__residual_blocks.append(residual_block)

    def fix_variables(self, *variables):
        for variable in variables:
            self.__fixed_variable_address.add(variable.__array_interface__['data'][0])

    def unfix_variables(self, *variables):
        for variable in variables:
            address = variable.__array_interface__['data'][0]
            if address in self.__fixed_variable_address:
                self.__fixed_variable_address.remove(address)

    def bound_variable(self, variable: np.ndarray, min=-np.inf, max=np.inf):
        """
        Lower and upper bounds on independent variable. Defaults to no bounds. 
        Note that the initial value of a bounded variable must lie in the boundary.
        :param variable: 
        :param min: array_like or scalar
            The array must match the size of variable or be a scalar. -np.inf means no bound.
        :param max: array_like or scalar
            The array must match the size of variable or be a scalar. np.inf means no bound.
        :return: None
        """
        address = variable.__array_interface__['data'][0]
        col_range = self.__address_col_range_map[address]
        assert np.isscalar(min) or min.shape[0] == col_range[1] - col_range[0]
        assert np.isscalar(max) or max.shape[0] == col_range[1] - col_range[0]
        self.__address_bounds_map[address] = (min, max)

    def solve(self, method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=None, x_scale='jac', verbose=1):
        """
        solve the problem.
        :param method: {'trf', 'dogleg', 'lm'}
            - trf: Trust Region Reflective algorithm, particular suitable for large sparse problem with bounds.
            - dogleg: dogleg algorithm with rectangle trust region, typical use case is small problems with bounds.
            - lm: Levenberg-Marquardt algorithm. Doesn't handle bounds and sparse jacobians. Suit for small problem.
        :param ftol: float
            Tolerance for termination by the change of the cost function. Default is 1e-8.
            The optimization process is stopped when dF < ftol * F, and there was an adequate agreement
            between a local quadratic model and the true model in the last step.
        :param xtol: float
            Tolerance for termination by the change of the independent variables. Default is 1e-8.
            The exact condition depends on the method used:
            - For ‘trf’ and ‘dogbox’ : norm(dx) < xtol * (xtol + norm(x))
            - For ‘lm’ : Delta < xtol * norm(xs), where Delta is a trust-region radius and xs is the value of x scaled according to x_scale parameter (see below).
        :param gtol:
            Tolerance for termination by the norm of the gradient. Default is 1e-8. The exact condition depends on a method used:
            - For ‘trf’ : norm(g_scaled, ord=np.inf) < gtol,
              where g_scaled is the value of the gradient scaled to account for the presence of the bounds [STIR].
            - For ‘dogbox’ : norm(g_free, ord=np.inf) < gtol,
              where g_free is the gradient with respect to the variables which are not in the optimal state on the boundary.
            - For ‘lm’ : the maximum absolute value of the cosine of angles
              between columns of the Jacobian and the residual vector is less than gtol, or the residual vector is zero.
        :param max_nfev: None or int
            Maximum number of function evaluations before the termination. If None (default), the value is chosen automatically:
            - For ‘trf’ and ‘dogbox’ : 100 * n.
            - For ‘lm’ : 100 * n if jac is callable and 100 * n * (n + 1) otherwise (because 'lm' counts function
             calls in Jacobian estimation).
        :param x_scale: array_like or 'jac'
            Characteristic scale of each variable. Setting x_scale is equivalent to reformulating the problem in scaled variables xs = x / x_scale.
            An alternative view is that the size of a trust region along j-th dimension is proportional to x_scale[j].
            Improved convergence may be achieved by setting x_scale such that a step of a given size along any of the
            scaled variables has a similar effect on the cost function. If set to ‘jac’,
            the scale is iteratively updated using the inverse norms of the columns of the Jacobian matrix
        :param verbose: {0, 1, 2}
             Level of algorithm’s verbosity:
             - 0 (default) : work silently.
             - 1 : display a termination report.
             - 2 : display progress during iterations (not supported by ‘lm’ method).
        :return: OptimizeResult
        """
        self.__initialize()
        res = scipy.optimize.least_squares(self.__batch_residual, self.__x0, self.__batch_jac, bounds=self.__bounds,
                                           method=method, ftol=ftol, xtol=xtol, gtol=gtol, x_scale=x_scale,
                                           loss=self.__batch_loss, tr_solver='lsmr',
                                           jac_sparsity=self.__jac_sparsity, max_nfev=max_nfev, verbose=verbose)
        self.__write_back_to_variables(res.x)
        return res

    def __batch_residual(self, x):
        """
        batch all residual functions to a single unified one.
        :param x:
        :return:
        """
        y = np.zeros(self.__dim_residual, dtype=np.float64)
        for residual_block in self.__residual_blocks:
            variables = []
            for col_range in residual_block.col_ranges:
                variables.append(x[col_range[0]: col_range[1]])
            row_range = residual_block.row_range
            y[row_range[0]: row_range[1]] = residual_block.residual_func(*variables)
        return y

    def __batch_jac(self, x):
        """
        batch all jacobian functions to a single unified one.
        :param x:
        :return:
        """
        J = scipy.sparse.dok_matrix((self.__dim_residual, self.__dim_variable), dtype=np.float64)
        for residual_block in self.__residual_blocks:
            variables = []
            for col_range in residual_block.col_ranges:
                variables.append(x[col_range[0]: col_range[1]])
            # calculate jacobian matrix at current state
            jac = residual_block.jac_func(*variables)
            # set jacobian matrix to the whole-problem jacobian.
            row_range = residual_block.row_range
            shift = 0
            for col_range in residual_block.col_ranges:
                dim = col_range[1] - col_range[0]
                J[row_range[0]: row_range[1], col_range[0]: col_range[1]] = jac[:, shift: shift + dim]
                shift += dim

        # mask out jacobian blocks of fixed variables
        for address in self.__fixed_variable_address:
            col_range = self.__address_col_range_map.get(address)
            if col_range is None:
                continue
            J[:, col_range[0]: col_range[1]] = 0

        return J

    def __batch_loss(self, z):
        """
        batch all loss functions to a single unified one.
        :param z: z = f(x)**2
        :return: an array_like with shape (3, m) where row 0 contains function values, 
        row 1 contains first derivatives and row 2 contains second derivatives.
        """
        assert z.shape[0] == self.__dim_residual
        loss = np.zeros((3, z.shape[0]), dtype=np.float64)
        for residual_block in self.__residual_blocks:
            row_range = residual_block.row_range
            loss[:, row_range[0]: row_range[1]] = residual_block.loss_func(z[row_range[0]: row_range[1]])
        return loss

    def __initialize(self):
        # set initial state
        self.__x0 = np.zeros(self.__dim_variable, dtype=np.float64)
        for col_range, x0 in self.__col_range_variable_map.items():
            self.__x0[col_range[0]: col_range[1]] = x0

        # set state bounds
        self.__bounds = [np.empty(self.__x0.shape, dtype=np.float64), np.empty(self.__x0.shape, dtype=np.float64)]
        self.__bounds[0][:] = -np.inf
        self.__bounds[1][:] = np.inf
        for address, bound in self.__address_bounds_map.items():
            col_range = self.__address_col_range_map[address]
            self.__bounds[0][col_range[0]: col_range[1]] = bound[0]
            self.__bounds[1][col_range[0]: col_range[1]] = bound[1]

        # set jacobian sparsity for the whole problem
        self.__jac_sparsity = scipy.sparse.dok_matrix((self.__dim_residual, self.__dim_variable), dtype=int)
        for residual_block in self.__residual_blocks:
            if residual_block.jac_sparsity is None:
                jac_sparsity = np.ones((residual_block.dim_residual, residual_block.dim_variable), dtype=int)
            else:
                jac_sparsity = residual_block.jac_sparsity

            row_range = residual_block.row_range
            shift = 0
            for col_range in residual_block.col_ranges:
                dim = col_range[1] - col_range[0]
                self.__jac_sparsity[row_range[0]: row_range[1], col_range[0]: col_range[1]] = \
                    jac_sparsity[:, shift: shift + dim]
                shift += dim

        # mask out jacobian blocks of fixed variables
        for address in self.__fixed_variable_address:
            col_range = self.__address_col_range_map.get(address)
            if col_range is None:
                continue
            self.__jac_sparsity[:, col_range[0]: col_range[1]] = 0

    def __write_back_to_variables(self, x):
        for col_range, variable in self.__col_range_variable_map.items():
            variable[:] = x[col_range[0]: col_range[1]]
