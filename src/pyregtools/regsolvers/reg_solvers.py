import numpy as np
#import scipy.io as scio
from scipy import linalg # for svd
from sklearn.linear_model import Ridge
#from scipy import optimize
import warnings
import cvxpy as cvx

def csvd(A):
    """ Compute the compact singular value decomposition (SVD).

    It is valid for matrices :math:`\\mathbf{A} \\in \\mathbb{C}^{M \\times L}`.
    For :math:`M \\geq L` (overdetermined system of equations), the decomposition is

    .. math::

        \\mathbf{A} = \\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}^{H}.

    For :math:`M < L` (underdetermined system of equations), the SVD is instead computed 
    from the Hermitian (conjugate and transpose) :math:`\\mathbf{A}^{H}`, as

    .. math::
    
        \\mathbf{A}^{H} = \\mathbf{V}\\boldsymbol{\\Sigma}\\mathbf{U}^{H},

    with the returned matrices are rearranged such that the output convention remains 
    consistent regardless of the shape of :math:`\\mathbf{A}`.

    Parameters
    ----------
    A : ndarray, shape (M, L)
        Sensing matrix, where :math:`M` is the number of measurements and
        :math:`L` is the number of unknowns.

    Returns
    -------
    u : ndarray
        Left singular vectors (not the Hermitian).
    sig : ndarray, shape (:math:`\\text{min}(M,L)`)
        Singular values in descending order.
    v : ndarray
        Right singular vectors (not the Hermitian).

    Notes
    -----
    The returned matrices satisfy

    .. math::

        \\mathbf{A} = \\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}^{H},

    where :math:`(\\cdot)^H` denotes the conjugate transpose.
    """
    M, L = A.shape
    if M >= L: # more measurements than unknowns
        u, sig, v = np.linalg.svd(A, full_matrices=False)
        v = np.conjugate(v.T)
    else:
        v, sig, u = np.linalg.svd(np.conjugate(A.T), full_matrices=False)
        u = np.conjugate(u.T)
    return u, sig, v

def gram_matrix(A):
    """ Computes Gram matrix of matrix :math:`\\mathbf{A} \\in \\mathbb{C}^{M \\times L}`.

    Matrix :math:`\\mathbf{A}` is first normalized by its column norms, resulting in
    :math:`\\mathbf{\\bar{A}}`. Then, the Gram matrix is 

    .. math::
        
            \\mathbf{G} = \\mathbf{\\bar{A}}^{H}\\mathbf{\\bar{A}},
    
    where :math:`\\mathbf{G}` scales from 0 to 1.

    That allows one to analyze if :math:`\\mathbf{A}` has correlated columns. The coherence
    is also computed as 
    
    .. math::
   
       \\mathrm{cohe} = \\max\\left(|\\mathbf{G}_{ij}|\\right).
    
    Parameters
    ----------
    A : ndarray, shape (M, L)
        Sensing matrix, where :math:`M` is the number of measurements and
        :math:`L` is the number of unknowns.

    Returns
    -------
    G : ndarray, shape (L, L)
        Gram matrix.
    cohe : float
        Matrix coherence.
    """
    # Compute L2 norm of each row vector
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    # Avoid division by zero for zero-columns
    col_norms[col_norms == 0] = 1.0 # numerical trick for zero-th norm cols.
    A_normalized = A / col_norms
    # Gram matrix
    gram_mtx = A_normalized.conj().T @ A_normalized
    # Take the absolute values
    abs_gram = np.abs(gram_mtx)
    # Fill the diagonal with zeros to satisfy i != j
    np.fill_diagonal(abs_gram, 0.0)
    # Find the maximum value
    cohe = np.max(abs_gram)
    return gram_mtx, cohe

def least_sq(A, b):
    """ Least squares solver.

    Computes the solution using numpy least-squares using :func:`numpy.linalg.lstsq`.

    For reference, the least-squares solution is

    .. math::
    
            \\mathbf{x} = (\\textbf{A}^H\\textbf{A})^{-1}\\textbf{A}^H \\textbf{b}.

    Parameters
    ----------
        A : ndarray, shape (M, L)
            Sensing matrix, where :math:`M` is the number of measurements and
            :math:`L` is the number of unknowns.
        b : ndarray, shape (M,)
            Measurement vector :math:`\\in \\mathbb{C}^{M}`.

    Returns
    -------
        x_lsq : ndarray, shape (L,)
            Solution vector :math:`\\in \\mathbb{C}^{L}`.
    """
    x_lsq = np.linalg.lstsq(A, b)[0]
    return x_lsq

def tikhonov(u,s,v,b,lambd_value):
    """ Computes the Tikhonov regularized solution [1]_ [2]_.
    
    This solver is based on the SVD of the sensing matrix 
    (:math:`\\mathbf{A} = \\mathbf{U} \\boldsymbol{\\Sigma} \\mathbf{V}^H`), with the
    solution given by

    .. math::
        
            \\mathbf{x}_{\\lambda} = \\textbf{V}(\\boldsymbol{\\Sigma}^{2}+\\lambda\\textbf{I})^{-1}\\boldsymbol{\\Sigma}\\textbf{U}^H \\textbf{b}.

    Parameters
    ----------
        u : ndarray
                Left singular vectors.
        sig : ndarray, shape (:math:`\\text{min}(M,L)`)
            Singular values in descending order.
        v : ndarray
            Right singular vectors.
        b : ndarray, shape (M,)
            Measurement vector :math:`\\in \\mathbb{C}^{M}`
        lambd_value : float
            Regularization parameter (:math:`\\geq 0`)

    Returns
    -------
        x_lambda : ndarray, shape (L,)
            Solution vector :math:`\\in \\mathbb{C}^{L}`.
    
    References
    ----------
    .. [1] P. C. Hansen, "Regularization Tools: A Matlab package for
        analysis and solution of discrete ill-posed problems,"
        Numerical Algorithms, vol. 6, pp. 1--35, 1994.

    .. [2] A. N. Tikhonov and V. Y. Arsenin, "Solutions of Ill-Posed
        Problems," Wiley, 1977.

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

def tikhonov_analytic(A,b,lambd_value):
    """ Computes the Tikhonov regularized solution.
        
    This solver is based on the analytic formula given by

    .. math::
        
            \\mathbf{x}_{\\lambda} = \\textbf{A}^H(\\textbf{A}\\textbf{A}^H+\\lambda\\textbf{I})^{-1}\\textbf{b}.

    Parameters
    ----------
        A : ndarray, shape (M, L)
            Sensing matrix, where :math:`M` is the number of measurements and
            :math:`L` is the number of unknowns.
        b : ndarray, shape (M,)
            Measurement vector :math:`\\in \\mathbb{C}^{M}`
        lambd_value : float
            Regularization parameter (:math:`\\geq 0`)

    Returns
    -------
        x_lambda : ndarray, shape (L,)
            Solution vector :math:`\\in \\mathbb{C}^{L}`.

    """ 
    #Hm = np.matrix(h_mtx)
    AH = A.conj().T
    x_lambda = AH @ np.linalg.inv(A @ AH +\
        (lambd_value**2)*np.identity(len(b))) @ b
    return x_lambda

def sklearn_ridge(A,b,lambd_value):
    """ Computes the Tikhonov regularized solution using :class:`~sklearn.linear_model.Ridge`.

    This particular solver is valid only for real matrices and data.

    Parameters
    ----------
        A : ndarray, shape (M, L)
            Sensing matrix :math:`\\in \\mathbb{R}^{M \\times L}`, 
            where :math:`M` is the number of measurements and
            :math:`L` is the number of unknowns.
        b : ndarray, shape (M,)
            Measurement vector :math:`\\in \\mathbb{R}^{M}`
        lambd_value : float
            Regularization parameter (:math:`\\geq 0`)

    Returns
    -------
        x_lambda : ndarray, shape (L,)
            Solution vector :math:`\\in \\mathbb{R}^{L}`.    
    """
    # Form a real H2 matrix and p2 measurement   
    #warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    regressor = Ridge(alpha=lambd_value, fit_intercept = False, solver = 'svd')
    x_lambda = regressor.fit(A, b).coef_
    return x_lambda

def sklearn_ridge_c(A,b,lambd_value):
    """ Computes the Tikhonov regularized solution using :class:`~sklearn.linear_model.Ridge`.
    
    This particular solver is valid for complex matrices and data. The problem is 
    reformulated to accomodate this need.

    Parameters
    ----------
        A : ndarray, shape (M, L)
            Sensing matrix :math:`\\in \\mathbb{C}^{M \\times L}`, 
            where :math:`M` is the number of measurements and
            :math:`L` is the number of unknowns.
        b : ndarray, shape (M,)
            Measurement vector :math:`\\in \\mathbb{C}^{M}`
        lambd_value : float
            Regularization parameter (:math:`\\geq 0`)

    Returns
    -------
        x_lambda : ndarray, shape (L,)
            Solution vector :math:`\\in \\mathbb{C}^{L}`.    
    """
    # Form a real H2 matrix and p2 measurement   
    #warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    H2 = np.vstack((np.hstack((A.real, -A.imag)),
        np.hstack((A.imag, A.real))))
    p2 = np.vstack((b.real,b.imag)).flatten()
    regressor = Ridge(alpha=lambd_value, fit_intercept = False, solver = 'svd')
    x2 = regressor.fit(H2, p2).coef_
    x_lambda = x2[:A.shape[1]]+1j*x2[A.shape[1]:]
    return x_lambda

def cvx_reg(A, b, lambd_value, is_lasso = False, is_complex = False):
    """ Computes the Tikhonov regularized solution using `CVXPY <https://www.cvxpy.org/>`_.
    
    This solver can perform both Ridge or Lasso regressions. It also can deal with 
    real or complex data. The options are specified by the user as booleans. 

    The optimization problem to solve is formulated as

    .. math::
            
        \\mathbf{\\tilde{x}} = \\underset{\\mathbf{x}}{\\operatorname{argmin}}
        \\left\\{
        \\|\\mathbf{A}\\mathbf{x} - \\mathbf{b}\\|_2^2
        +
        \\lambda \\|\\mathbf{x}\\|_{\\mathcal{\\ell_n}}
        \\right\\},

    where :math:`\\|\\mathbf{x}\\|_{\\ell_n}` is either 

    - :math:`\\|\\mathbf{x}\\|_{2}`: the :math:`\\ell_2` solution norm (Ridge regression) or,
    - :math:`\\|\\mathbf{x}\\|_{1}`:  the :math:`\\ell_1` solution norm (Lasso sparse regression).
    
    Parameters
    ----------
        A : ndarray, shape (M, L)
            Sensing matrix :math:`\\in \\mathbb{C}^{M \\times L}`, 
            where :math:`M` is the number of measurements and
            :math:`L` is the number of unknowns.
        b : ndarray, shape (M,)
            Measurement vector :math:`\\in \\mathbb{C}^{M}`.
        lambd_value : float
            Regularization parameter (:math:`\\geq 0`).
        is_lasso : bool
            Type of regression. If is_lasso is False, then use l_norm = 2 
            for a Ridge regression. If is_lasso is True, then use l_norm = 1 
            for a Lasso regression.
        is_complex : bool
            Whether the measurement is complex or not.

    Returns
    -------
        x_lambda : ndarray, shape (L,)
            Solution vector :math:`\\in \\mathbb{C}^{L}`.    
    """
    if lambd_value < 0:
        warnings.warn("Illegal regularization parameter lambda. I'll set it to 1.0")
        lambd_value = 1.0
    if is_lasso:
        l_norm = 1
    else:
        l_norm = 2
    # Create variable to be solved for.
    m, l = A.shape
    x = cvx.Variable(shape = l, complex = is_complex)      
    # Form objective.
    obj = cvx.Minimize(cvx.norm(A @ x - b, 2) + (lambd_value)*cvx.norm(x, l_norm))
    # Form and solve problem.
    prob = cvx.Problem(obj)
    prob.solve();
    return x.value

def tsvd(u,s,v,b,k):
    """ Computes the Truncated SVD regularized solution [1]_.
    
    This solver is based on the SVD of the sensing matrix 
    (:math:`\\mathbf{A} = \\mathbf{U} \\boldsymbol{\\Sigma} \\mathbf{V}^H`), with the
    solution given by

    .. math::
        
            \\mathbf{x}_{k} = \\sum\\limits_{i = 1}^{k} \\frac{\\textbf{u}_i^H \\textbf{b}}{\\sigma_i}\\textbf{v}_i.

    where :math:`\\textbf{u}_i` and :math:`\\textbf{v}_i` are the i-th columns of the 
    :math:`\\textbf{U}` and :math:`\\textbf{V}` matrices; :math:`\\sigma_i` is the i-th 
    singular value, and :math:`k` is the number of singular values to include.
    
    Parameters
    ----------
        u : ndarray
                Left singular vectors.
        sig : ndarray, shape (:math:`\\text{min}(M,L)`)
            Singular values in descending order.
        v : ndarray
            Right singular vectors.
        b : ndarray, shape (M,)
            Measurement vector :math:`\\in \\mathbb{C}^{M}`
        k : int
            Number of singular values to include.

    Returns
    -------
        x_k : ndarray, shape (L,)
            Solution vector :math:`\\in \\mathbb{C}^{L}`.
    
    References
    ----------
    .. [1] P. C. Hansen, "Regularization Tools: A Matlab package for
        analysis and solution of discrete ill-posed problems,"
        Numerical Algorithms, vol. 6, pp. 1--35, 1994.
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
    """ Computes the Selective SVD regularized solution [1]_.
    
    This solver is based on the SVD of the sensing matrix 
    (:math:`\\mathbf{A} = \\mathbf{U} \\boldsymbol{\\Sigma} \\mathbf{V}^H`), with the
    solution given by

    .. math::
        
            \\mathbf{x}_{\\tau} = \\sum\\limits_{|\\textbf{u}_i^H \\textbf{b}|>\\tau} \\frac{\\textbf{u}_i^H \\textbf{b}}{\\sigma_i}\\textbf{v}_i.

    where :math:`\\textbf{u}_i` and :math:`\\textbf{v}_i` are the i-th columns of the 
    :math:`\\textbf{U}` and :math:`\\textbf{V}` matrices; :math:`\\sigma_i` is the i-th 
    singular value, and :math:`\\tau` is the threshold of singular values inclusion.
    
    Parameters
    ----------
        u : ndarray
                Left singular vectors.
        sig : ndarray, shape (:math:`\\text{min}(M,L)`)
            Singular values in descending order.
        v : ndarray
            Right singular vectors.
        b : ndarray, shape (M,)
            Measurement vector :math:`\\in \\mathbb{C}^{M}`.
        tau : float
            Threshhold of singular value inclusion.

    Returns
    -------
        x_k : ndarray, shape (L,)
            Solution vector :math:`\\in \\mathbb{C}^{L}`.
    
    References
    ----------
    .. [1] P. C. Hansen, "Regularization Tools: A Matlab package for
        analysis and solution of discrete ill-posed problems,"
        Numerical Algorithms, vol. 6, pp. 1--35, 1994.
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
    """ Computes the regularized solution by constrained optmization using `CVXPY <https://www.cvxpy.org/>`_.
    
    This solver can perform both the usual regularized solution promoting a continuous and
    smooth solution, or compute a solution considering sparsity (compressed sensing). 
    It also can deal with real or complex data. The options are specified by the user 
    as booleans. 

    The optimization problem to solve is formulated as

    .. math::
            
        \\mathbf{\\tilde{x}} = \\underset{\\mathbf{x}}{\\operatorname{argmin}}
        \\left\\{\\|\\mathbf{x}\\|_{\\mathcal{\\ell_n}}\\right\\} \\qquad \\text{s.t.} \\qquad 
        \\|\\mathbf{A}\\mathbf{x} - \\mathbf{b}\\|_2^2 \\geq \\|\\mathbf{n}\\|_2^2,

    where :math:`\\|\\mathbf{x}\\|_{\\ell_n}` is either 

    - :math:`\\|\\mathbf{x}\\|_{2}`: the :math:`\\ell_2` solution norm (smooth solution) or,
    - :math:`\\|\\mathbf{x}\\|_{1}`:  the :math:`\\ell_1` solution norm (sparse solution).
    
    Parameters
    ----------
        A : ndarray, shape (M, L)
            Sensing matrix :math:`\\in \\mathbb{C}^{M \\times L}`, 
            where :math:`M` is the number of measurements and
            :math:`L` is the number of unknowns.
        b : ndarray, shape (M,)
            Measurement vector :math:`\\in \\mathbb{C}^{M}`.
        noise_norm : float
            Estimation of the noise vector norm (:math:`\\geq 0`).
        is_cs : bool
            Type of regression. If is_cs is False, then use l_norm = 2 
            for a l2 regression. If is_cs is True, then use l_norm = 1 
            for a compressed sensing regression.
        is_complex : bool
            Whether the measurement is complex or not.

    Returns
    -------
        x_lambda : ndarray, shape (L,)
            Solution vector :math:`\\in \\mathbb{C}^{L}`.    
    """
    #    .. math::
    #   min(|x|_{l_norm}|), s.t. |Ax-b|_2^2 <= |n|_2^2
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
    """ Deprecated
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