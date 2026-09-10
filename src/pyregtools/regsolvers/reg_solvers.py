import numpy as np
#import scipy.io as scio
from scipy import linalg # for svd
from sklearn.linear_model import Ridge
#from scipy import optimize
import warnings
import cvxpy as cvx


def least_sq(h_mtx, bm):
    """ Least squares solver

    Parameters
    ----------
        h_mtx : numpy ndarray
            sensing matrix
        b: numpy 1darray
            your measurement vector (size: Nm x 1)
    Returns
    -------
        x_lsq : numpy 1darray
            estimated solution to inverse problem
    """
    x_lsq = np.linalg.lstsq(h_mtx, bm)[0]
    return x_lsq

def tikhonov(u,s,v,b,lambd_value):
    """ Computes the Tikhonov regularized solution x_lambda, given the SVD
    
    Based on the matlab routine by: Per Christian Hansen, DTU Compute, April 14, 2003.
    Reference: A. N. Tikhonov & V. Y. Arsenin, "Solutions of Ill-Posed
    Problems", Wiley, 1977.

    Parameters
    ----------
        u : numpy ndarray
            left singular vectors
        sig : numpy 1darray
            singular values
        v : numpy ndarray
            right singular vectors
        b: numpy 1darray
            your measurement vector (size: Nm x 1)
        lambd_value : float
            optimal regularization parameter
    Returns
    -------
        x_lambda : numpy 1darray
            estimated solution to inverse problem
    """
    # warn that lambda should be bigger than 0
    if lambd_value < 0:
        warnings.warn("Illegal regularization parameter lambda. I'll set it to 1.0")
        lambd_value = 1.0
    p = len(s)
    beta = np.conjugate(u[:,0:p]).T @ b
    zeta = s * beta
    x_lambda = v[:,0:p] @ np.divide(zeta, s**2 + lambd_value**2)
    return x_lambda

def tikhonov_analytic(h_mtx,bm,lambd_value):
    """ Computes the Tikhonov regularized solution x_lambda from analytical formula.

    Parameters
    ----------
        h_mtx : numpy ndarray
            sensing matrix
        bm: numpy 1darray
            your measurement vector (size: Nm x 1)
        lambd_value : float
            optimal regularization parameter
    Returns
    -------
        x_lambda : numpy 1darray
            estimated solution to inverse problem
    """
    #Hm = np.matrix(h_mtx)
    h_mtx_H = h_mtx.conj().T
    x_lambda = h_mtx_H @ np.linalg.inv(h_mtx @ h_mtx_H +\
                                         (lambd_value**2)*np.identity(len(bm))) @ bm
    return x_lambda

def sklearn_ridge(h_mtx,bm,lambd_value):
    """ Computes the Tikhonov regularized solution x_lambda using sklearn Ridge regression. 

    This particular setup is valid for real measurements.

    Parameters
    ----------
        h_mtx : numpy ndarray
            sensing matrix
        bm: numpy 1darray
            your measurement vector (size: Nm x 1)
        lambd_value : float
            optimal regularization parameter
    Returns
    -------
        x_lambda : numpy 1darray
            estimated solution to inverse problem
    """
    # Form a real H2 matrix and p2 measurement   
    #warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    regressor = Ridge(alpha=lambd_value, fit_intercept = False, solver = 'svd')
    x_lambda = regressor.fit(h_mtx, bm).coef_
    return x_lambda

def sklearn_ridge_c(h_mtx,bm,lambd_value):
    """ Computes the Tikhonov regularized solution x_lambda using sklearn Ridge regression. 

    This particular setup is valid for complex measurements. The problem is reformulated to
    accomodate this need.

    Parameters
    ----------
        h_mtx : numpy ndarray
            sensing matrix
        bm: numpy 1darray
            your measurement vector (size: Nm x 1)
        lambd_value : float
            optimal regularization parameter
    Returns
    -------
        x_lambda : numpy 1darray
            estimated solution to inverse problem
    """
    # Form a real H2 matrix and p2 measurement   
    #warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    H2 = np.vstack((np.hstack((h_mtx.real, -h_mtx.imag)),
        np.hstack((h_mtx.imag, h_mtx.real))))
    p2 = np.vstack((bm.real,bm.imag)).flatten()
    regressor = Ridge(alpha=lambd_value, fit_intercept = False, solver = 'svd')
    x2 = regressor.fit(H2, p2).coef_
    x_lambda = x2[:h_mtx.shape[1]]+1j*x2[h_mtx.shape[1]:]
    return x_lambda

def cvx_reg(A, b, lam, is_lasso = False, is_complex = False):
    """ Solves Ridge regression (tikhonov) or Lasso regression using cvx.

    This setup is valid for real or complex measurements. 
    If is_lasso = False, then the Ridge regression is solved. 
    If is_lasso = True, then the Lasso regression is solved.

    Parameters
    ----------
        A : numpy ndarray
            sensing matrix (MxL)
        b: numpy 1darray
            your measurement vector (size: M x 1)
        lam : float
            Regularization parameter.
        is_lasso : bool
            Type of regression. If is_lasso is False, then use l_norm = 2 
            for a Ridge regression. If is_lasso is True, then use l_norm = 1 
            for a Lasso regression
        is_complex : bool
            whether the measurement is complex or not 
    Returns
    -------
        x.value : numpy 1darray
            estimated solution to inverse problem
    """
    if lam < 0:
        warnings.warn("Illegal regularization parameter lambda. I'll set it to 1.0")
        lam = 1.0
    if is_lasso:
        l_norm = 1
    else:
        l_norm = 2
    # Create variable to be solved for.
    m, l = A.shape
    x = cvx.Variable(shape = l, complex = is_complex)      
    # Form objective.
    obj = cvx.Minimize(cvx.norm(A @ x - b, 2) + (lam)*cvx.norm(x, l_norm))
    # Form and solve problem.
    prob = cvx.Problem(obj)
    prob.solve();
    return x.value

def tsvd(u,s,v,b,k):
    """ Estimates truncated SVD regularized solution
    
    Parameters
    ----------
        u : numpy ndarray
            left singular vectors from csvd
        s : numpy 1darray
            singular values from csvd
        v : numpy ndarray
            right singular vectors from csvd
        b : numpy 1darray
            measured vector
        k : int
            number of singular values to include
    Returns
    -------
        x_k : numpy 1darray
            estimated solution to inverse problem
    """
    n,p = v.shape
    if k > p:
      warnings.warn('Illegal truncation parameter k. Setting k = p')
      k = p
    beta = np.conj(u[:,0:p]).T @ b
    xi = beta/s    
    x_k = v[:,0:k] @ xi[0:k]
    return x_k

def ssvd(u,s,v,b,tau):
    """ Estimates selective SVD regularized solution
    
    Parameters
    ----------
        u : numpy ndarray
            left singular vectors from csvd
        s : numpy 1darray
            singular values from csvd
        v : numpy ndarray
            right singular vectors from csvd
        s : numpy 1darray
            measured vector
        tau : float
            Threshhold
    Returns
    -------
        x_k : numpy 1darray
            estimated solution to inverse problem
    """
    n,p = v.shape
        
    beta_full = np.conj(u).T @ b
    idbeta = np.where(np.abs(beta_full) > tau)[0]
    beta = beta_full[idbeta]
    xi = beta/s[idbeta]
    v = v[:,idbeta]    
    x_tau = v @ xi
    return x_tau
 
def cvx_constrained(A, b, noise_norm, is_cs = False, is_complex = False):
    """ Solves regularized problem by constrained optmization.

    The following problem is solved
    .. math::
       min(|x|_{l_norm}|), s.t. |Ax-b|_2^2 <= |n|_2^2

    Parameters
    ----------
        A : numpy ndarray
            sensing matrix (MxL)
        b: numpy 1darray
            your measurement vector (size: M x 1)
        noise_norm : float
            norm of the noise (to set constraint)
        is_cs : bool
            Type of regression. If is_cs is False, then use l_norm = 2 
            for a l2 regression. If is_cs is True, then use l_norm = 1 
            for a compressed sensing regression
        is_complex : bool
            whether the measurement is complex or not 
    Returns
    -------
        x.value : numpy 1darray
            estimated solution to inverse problem
    """
    if noise_norm < 0:
        warnings.warn("Illegal noise norm. Must be larger than 0. I'll set it to 0.1")
        noise_norm = 0.1
    if is_cs:
        l_norm = 1
    else:
        l_norm = 2
    # Create variable to be solved for.
    m, l = A.shape
    x = cvx.Variable(shape = l, complex = is_complex, value = np.zeros(l))
    # Create constraint.
    constraints = [cvx.pnorm(A @ x - b, p = 2) <= noise_norm]
    # Form objective.
    obj = cvx.Minimize(cvx.norm(x, l_norm))
    # Form and solve problem.
    prob = cvx.Problem(obj, constraints)
    prob.solve();
    return x.value

def cvx_solver_c(A, b, noise_norm, l_norm = 2):
    """ Solves regularized problem by convex optmization.

    Parameters
    ----------
        A : numpy ndarray
            sensing matrix (MxL)
        b: numpy 1darray
            your measurement vector (size: M x 1)
        noise_norm : float
            norm of the noise (to set constraint)
        l_norm : int
            Type of norm to minimize x
    Returns
    -------
        x : numpy 1darray
            estimated solution to inverse problem
    """
    # Create variable to be solved for.
    m, l = A.shape
    x = cvx.Variable(shape = l, complex = True, value = np.zeros(l))
    
    # Create constraint.
    constraints = [cvx.pnorm(b - cvx.matmul(A, x), p=2) <= noise_norm]
    # Form objective.
    obj = cvx.Minimize(cvx.pnorm(x, p = l_norm))  
    # Form and solve problem.
    prob = cvx.Problem(obj, constraints)
    prob.solve();
    return x.value

