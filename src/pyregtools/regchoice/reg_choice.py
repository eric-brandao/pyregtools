"""
This module provides methods for automatically selecting the Tikhonov regularization
parameter :math:`\\lambda` in discrete inverse problems.

The parameter-choice methods implemented in this module are currently focused on selecting 
the regularization parameter :math:`\\lambda` for Tikhonov regularization. 
Tikhonov regularization provides a smooth filtering of the singular-value components 
and is one of the most widely used approaches for stabilizing discrete ill-posed 
inverse problems [2]_. In contrast, the selection of the discrete truncation parameter 
used in truncated singular value decomposition (TSVD) is not currently implemented. 
Support for automatic selection of the TSVD truncation parameter may be included in future 
versions of PyRegTools.

The implemented criteria are:

* discrepancy principle (DP),
* L-curve criterion,
* generalized cross-validation (GCV), and
* normalized cumulative periodogram (NCP).

The implementations are largely based on the algorithms described in
Hansen's Regularization Tools for MATLAB [1]_ and on the numerical
treatment of discrete inverse problems presented in [2]_.

Most parameter-choice methods implemented here operate on the singular
value decomposition (SVD) of the forward matrix

.. math::

    \\mathbf{A}
    =
    \\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}^{H},

together with the measurement vector :math:`\\mathbf{b}`. The projected
data coefficients are given by

.. math::

    \\boldsymbol{\\beta} = \\mathbf{U}^{H}\\mathbf{b}.

For methods requiring numerical optimization of a criterion
:math:`R(\\lambda)`, the criterion is first evaluated over a discrete,
logarithmically spaced set of regularization parameters. The range is
determined from the singular values of :math:`\\mathbf{A}`. The best
candidate from this discrete search is then used to define a bounded
interval in which a continuous optimization is performed.

This two-stage procedure provides an initial exploration of the
regularization-parameter space followed by a refined estimate of the
selected parameter.

Differently from the implementation in Ref. [1]_, PyRegTools allows the 
user to specify a lower bound for the regularization parameter :math:`\\lambda`, 
defined as a fraction of the largest singular value. This additional constraint 
reduces the susceptibility of the parameter-choice methods to spurious local minima 
or maxima of :math:`R(\\lambda)` occurring at values of :math:`\\lambda` that are 
too small to provide meaningful regularization.


References
----------
.. [1] P. C. Hansen, "Regularization Tools: A Matlab package for
       analysis and solution of discrete ill-posed problems,"
       Numerical Algorithms, vol. 6, pp. 1--35, 1994.

.. [2] P. C. Hansen, Discrete Inverse Problems: Insight and Algorithms,
       SIAM, Philadelphia, 2010.
"""

import numpy as np
import matplotlib.pyplot as plt
#import scipy.io as scio
#from scipy import linalg # for svd
#from sklearn.linear_model import Ridge
from scipy import optimize
import warnings

def get_regpar(s_valid, npoints = 200, r_min = 16 * np.finfo(float).eps):
    """
    Generate an initial search grid for the regularization parameter.

    Constructs a logarithmically spaced grid of candidate Tikhonov
    regularization parameters :math:`\\lambda`. The grid spans from the
    largest singular value to a lower bound determined by the larger of
    the smallest retained singular value and a prescribed fraction of
    the largest singular value.

    The lower bound is defined as

    .. math::

        \\lambda_{\\min}
        =
        \\max\\left(
        \\sigma_{\\min},
        \\sigma_{\\max}\\,r_{\\min}
        \\right),

    where :math:`r_{\\min}` is given by ``r_min``. The candidate
    regularization parameters are logarithmically spaced between
    :math:`\\sigma_{\\max}` and :math:`\\lambda_{\\min}`.

    Parameters
    ----------
    s_valid : ndarray, shape (p,)
        Valid range of singular values, ordered from largest to smallest.
        Typically, ``p`` corresponds to the number of singular values
        retained for the parameter-choice procedure.
    npoints : int, default=200
        Number of regularization parameters in the search grid.
    r_min : float, default=16 * eps
        Minimum regularization parameter relative to the largest singular
        value. Here, ``eps`` denotes machine precision for a floating-point
        number. The lower end of the search grid cannot be smaller than
        ``s_valid[0] * r_min``.

    Returns
    -------
    reg_par_grid : ndarray, shape (npoints,)
        Logarithmically spaced regularization-parameter grid, ordered from
        largest to smallest.
    """
    last_val = max(s_valid[-1], s_valid[0] * r_min)
    reg_par_grid = np.geomspace(s_valid[0], last_val, npoints)
    return reg_par_grid

def get_bounds(vec_to_minimize, reg_par_vec):
    """
    Determine bounds and tolerance for continuous optimization.

    Identifies the minimum of a criterion evaluated over an initial
    discrete grid of regularization parameters and defines a local search
    interval around the corresponding candidate parameter. The interval
    is subsequently used to refine the regularization parameter through
    bounded continuous optimization.

    The lower and upper bounds are taken from the neighboring grid points
    around the discrete minimum. If the minimum occurs at either endpoint
    of the grid, the corresponding endpoint is retained as a bound. The
    ``tolerance`` adapts from such values and will be an input for the 
    continuous optimization routines.

    Parameters
    ----------
    vec_to_minimize : ndarray, shape (npoints,)
        Values of the parameter-choice criterion evaluated at each
        regularization parameter in ``reg_par_vec``.
    reg_par_vec : ndarray, shape (npoints,)
        Initial grid of regularization parameters (Logarithmically spaced), 
        ordered from largest to smallest.

    Returns
    -------
    lower_bound : float
        Lower bound for the continuous optimization.
    upper_bound : float
        Upper bound for the continuous optimization.
    tolerance : float
        Absolute tolerance used by the continuous optimization routine.
        It is computed as the minimum of ``lower_bound / 50``, ``upper_bound / 50``,
        and ``1e-5``.

    Notes
    -----
    This function provides the transition between the initial discrete
    exploration of the regularization-parameter space and the subsequent
    continuous optimization of the parameter-choice criterion.
    """
    # find index of minimum (discrete curve)
    npoints = len(reg_par_vec)
    min_id = np.argmin(vec_to_minimize)
    # lower and upper bounds, and tolerance for continous optimization.
    lower_bound = reg_par_vec[int(np.amin(np.array([min_id+1, npoints-1])))]
    upper_bound = reg_par_vec[int(np.amax([min_id - 1, 0]))]
    tolerance = np.amin([lower_bound/50, upper_bound/50, 1e-5])
    return lower_bound, upper_bound, tolerance    

def f_eta_rho(reg_param, s, xi, beta, beta_perp_sq, M, L):
    """
    Computes the Tikhonov filter factors for each regularization parameter
    :math:`\\lambda` as [1]_.

    .. math::

        \\varphi_i(\\lambda)
        =
        \\frac{\\sigma_i^2}
        {\\sigma_i^2 + \\lambda^2},

    where :math:`\\sigma_i` are the singular values of the forward matrix.
    The corresponding solution and residual norms are

    .. math::

        \\eta(\\lambda)
        =
        \\|\\mathbf{x}_{\\lambda}\\|_2,

    and

    .. math::

        \\rho(\\lambda)
        =
        \\|\\mathbf{A}\\mathbf{x}_{\\lambda} - \\mathbf{b}\\|_2.

    Using the SVD coefficients, these quantities are evaluated from the
    filter factors without explicitly computing the regularized solution
    for every value of :math:`\\lambda`.

    Parameters
    ----------
    reg_param : ndarray, shape (npoints,)
        Regularization parameters :math:`\\lambda` at which the filter
        factors and norms are evaluated.
    s : ndarray, shape (p,)
        Singular values :math:`\\sigma_i` of the forward matrix.
    xi : ndarray, shape (p,)
        SVD solution coefficients, given by
        :math:`\\xi_i = \\beta_i / \\sigma_i`.
    beta : ndarray, shape (p,)
        Projected measurement coefficients
        :math:`\\boldsymbol{\\beta} = \\mathbf{U}^H\\mathbf{b}`.
    beta_perp_sq : float
        Squared norm of the component of the measurement vector outside
        the column space represented by :math:`\\mathbf{U}`, computed as
        :math:`\\|\\mathbf{b}\\|_2^2 -
        \\|\\mathbf{U}^H\\mathbf{b}\\|_2^2`.
    M : int
        Number of measurements.
    L : int
        Number of unknowns.

    Returns
    -------
    eta : ndarray, shape (npoints,)
        Solution norm :math:`\\eta(\\lambda) =
        \\|\\mathbf{x}_{\\lambda}\\|_2` for each regularization parameter.
    rho : ndarray, shape (npoints,)
        Residual norm :math:`\\rho(\\lambda) =
        \\|\\mathbf{A}\\mathbf{x}_{\\lambda}-\\mathbf{b}\\|_2` for each
        regularization parameter.

    Notes
    -----
    For overdetermined problems (:math:`M > L`), the compact SVD does not
    span the entire measurement space. The residual therefore contains a
    component orthogonal to the column space of :math:`\\mathbf{A}` that is
    not included in :math:`\\|(1-\\varphi_i)\\beta_i\\|_2`.

    The squared norm of this component is

    .. math::
    
            \\beta_{\\perp}^{2}
            =
            \\max\\left(
            \\|\\mathbf{b}\\|_2^2
            -
            \\|\\boldsymbol{\\beta}\\|_2^2,
            0
            \\right),
    
    where the maximum with zero prevents small negative values caused by
    floating-point roundoff. The complete residual norm is therefore computed as

    .. math::

        \\rho(\\lambda)
        =
        \\sqrt{
        \\sum_i (1-\\varphi_i)^2 |\\beta_i|^2 + \\beta_2
        }.

    This correction is required only for overdetermined problems.

    References
    ----------
    .. [1] P. C. Hansen, *Discrete Inverse Problems: Insight and
           Algorithms*, SIAM, Philadelphia, 2010.
    """
    eta = np.zeros(len(reg_param))
    rho = np.zeros(len(reg_param))
    for i, lam in enumerate(reg_param):
        f = s**2 / (s**2 + lam ** 2) # filter factors
        eta[i] = np.linalg.norm(f * xi) # solution norm
        rho[i] = np.linalg.norm((1-f) * beta) # residual norm
    if (M > L and beta_perp_sq > 0):
        rho = np.sqrt(rho ** 2 + beta_perp_sq)
    return eta, rho

def complement_filter_factors(lambda_val, s, check_s_sq = False):
    """
    Compute the complement of the Tikhonov filter factors.

    For a regularization parameter :math:`\\lambda` and singular values
    :math:`\\sigma_i`, computes

    .. math::

        1 - \\varphi_i
        =
        \\frac{\\lambda^2}
        {\\sigma_i^2 + \\lambda^2},

    where :math:`\\varphi_i` are the Tikhonov filter factors.

    Parameters
    ----------
    lambda_val : float
        Regularization parameter :math:`\\lambda`.
    s : ndarray, shape (p,)
        Singular values :math:`\\sigma_i`. If ``check_s_sq=True``,
        ``s`` must instead contain the squared singular values
        :math:`\\sigma_i^2`.
    check_s_sq : bool, default=False
        If ``True``, assumes that both the singular values and the
        regularization parameter are already squared, avoiding their
        repeated computation.

    Returns
    -------
    cf : ndarray, shape (p,)
        Complement of the Tikhonov filter factors, :math:`1-\\varphi_i`.
    """
    if not check_s_sq:
        cf = (lambda_val ** 2) / (s ** 2 + lambda_val ** 2)
    else:
        cf = lambda_val / (s + lambda_val)
    return cf

def l_curve(u, s, b, r_min = 16 * np.finfo(float).eps, 
            plotit = False, plot_in_color = True):
    """
    Select the Tikhonov regularization parameter using the L-curve criterion.

    Computes the corner of the L-curve and returns the corresponding
    regularization parameter. The search is performed over an initial
    logarithmically spaced grid of candidate values and refined using the
    L-curve curvature.

    Optionally, the function plots the L-curve together with the selected
    corner and the curvature evaluated over the initial search grid.

    Parameters
    ----------
    u : ndarray
        Matrix containing the left singular vectors of the forward matrix.
    s : ndarray, shape (p,)
        Singular values, ordered from largest to smallest.
    b : ndarray, shape (M,)
        Measurement vector.
    r_min : float, default=16 * eps
        Minimum regularization parameter relative to the largest singular
        value. The lower bound of the initial search grid is constrained by
        ``s[0] * r_min``, where ``eps`` denotes floating-point machine
        precision.
    plotit : bool, default=False
        If ``True``, plot the L-curve and the curvature used to identify
        its corner.
    plot_in_color : bool, default=True
        If ``True``, use colored lines and markers in the plot. If ``False``,
        use a grayscale representation. This parameter is only used when
        ``plotit=True``.

    Returns
    -------
    lam_opt : float
        Tikhonov regularization parameter corresponding to the estimated
        corner of the L-curve.

    Notes
    -----
    The initial regularization-parameter grid is generated using
    ``get_regpar``. The L-curve corner and the associated residual and
    solution norms are then computed using ``l_corner_new``.
    """
    # Sizes and shapes
    npoints = 200  # Number of points on the L-curve    
    #p = len(s)
    # Initial search grid
    reg_param = get_regpar(s_valid = s, npoints = npoints, r_min = r_min)
    # Compute everything
    lam_opt, curv, rho_c, eta_c, eta, rho = l_corner(reg_param,u,s,b)
    # want to plot the L curve?
    if plotit:
        if plot_in_color:
            color_dict = dict(l_c = 'dodgerblue', lam_l = 'r', c_c = 'navy')
        else:
            color_dict = dict(l_c = 'k', lam_l = 'grey', c_c = 'k')
        fig = plt.figure(figsize = (6,3))
        # L-curve
        plt.loglog(rho, eta, color = color_dict['l_c'], linewidth = 1.5)
        plt.loglog(rho_c, eta_c, marker = 'o', color = color_dict['lam_l'],
                   markerfacecolor = 'none')
        plt.xlim((10**np.floor(np.log10(rho.min())), 10**np.ceil(np.log10(rho.max()))))
        plt.ylim((10**np.floor(np.log10(eta.min())), 10**np.ceil(np.log10(eta.max()))))

        plt.vlines(x = rho_c, ymin=plt.ylim()[0], ymax = eta_c, color=color_dict['lam_l'], 
                   linestyle=':', linewidth = 0.7)
        plt.hlines(y = eta_c, xmin=plt.xlim()[0], xmax = rho_c, color=color_dict['lam_l'], 
                   linestyle=':', linewidth = 0.7)
        plt.title(r'L-curve ($\lambda = ${:.6f})'.format(lam_opt), loc = 'right')
        plt.xlabel(r'Residual norm $||Ax - b||_2$')
        plt.ylabel(r'Solution norm $||x||_2$')
        plt.grid(linestyle = '--', which='both')
        plt.tight_layout()
        # Curvature
        ax2 = fig.add_axes([0.60, 0.55, 0.3, 0.3])
        ax2.semilogx(reg_param, -curv, color = color_dict['c_c'], linewidth = 1.5)
        ax2.semilogx(lam_opt, np.amax(-curv), marker = 'o', color = color_dict['lam_l'],
                   markerfacecolor = 'none')
        ax2.set_xlim((reg_param.min(), reg_param.max()))
        ax2.set_ylim((-0.1*(1.2*np.amax(-curv)), 1.2*np.amax(-curv)))
        ax2.vlines(x = lam_opt, ymin=ax2.set_ylim()[0], ymax = np.amax(-curv), 
                   color=color_dict['lam_l'], linestyle=':', linewidth = 0.7)
        ax2.hlines(y = np.amax(-curv), xmin=ax2.set_xlim()[0], xmax = lam_opt, 
                   color=color_dict['lam_l'], linestyle=':', linewidth = 0.7)
        ax2.grid(linestyle = '--')
        ax2.set_xlabel(r'$\lambda$')
        ax2.set_ylabel(r'$-c(\lambda)$')
    return lam_opt

def l_corner(reg_param,u,s,b):
    """
    Estimate the corner of the L-curve [1]_, [2]_.

    Determines the Tikhonov regularization parameter associated with the
    point of maximum curvature of the L-curve. In other words, it finds
    the corner of the L-curve (balancing the solution and the residual norms).
    The function first evaluates the residual norm and solution norm over a discrete 
    grid of candidate regularization parameters and then refines the estimate 
    using a bounded one-dimensional optimization of the L-curve curvature.

    The L-curve is represented by

    .. math::

        \\rho(\\lambda)
        =
        \\|\\mathbf{A}\\mathbf{x}_{\\lambda} - \\mathbf{b}\\|_2,

    and

    .. math::

        \\eta(\\lambda)
        =
        \\|\\mathbf{x}_{\\lambda}\\|_2,

    evaluated for the values of :math:`\\lambda` contained in
    ``reg_param``. The corner is identified as the point of maximum
    curvature in the :math:`(\\log(\\rho), \\log(\\eta))` plane.

    Parameters
    ----------
    reg_param : ndarray, shape (npoints,)
        Candidate Tikhonov regularization parameters used for the initial
        discrete evaluation of the L-curve.
    u : ndarray, shape (M, p)
        Matrix containing the left singular vectors of the forward matrix.
    s : ndarray, shape (p,)
        Singular values, ordered from largest to smallest.
    b : ndarray, shape (M,)
        Measurement vector.

    Returns
    -------
    reg_c : float
        Regularization parameter corresponding to the estimated corner of
        the L-curve.
    curv : ndarray, shape (npoints,)
        Negative L-curve curvature evaluated at the candidate values in
        ``reg_param``.
    rho_c : float
        Residual norm evaluated at ``reg_c``.
    eta_c : float
        Solution norm evaluated at ``reg_c``.
    eta : ndarray, shape (npoints,)
        Solution norm evaluated over the initial regularization-parameter
        grid.
    rho : ndarray, shape (npoints,)
        Residual norm evaluated over the initial regularization-parameter
        grid.

    Notes
    -----
    The measurement vector is first projected onto the left singular-vector
    basis,

    .. math::

        \\boldsymbol{\\beta}
        =
        \\mathbf{U}^{H}\\mathbf{b},

    and the corresponding unregularized SVD coefficients are computed as

    .. math::

        \\xi_i
        =
        \\frac{\\beta_i}{\\sigma_i}.

    The L-curve quantities :math:`\\rho(\\lambda)` and :math:`\\eta(\\lambda)`
    are first evaluated on the discrete grid ``reg_param``. The negative
    curvature is then evaluated at each grid point. A local search interval
    surrounding the best discrete candidate is obtained with ``get_bounds``,
    after which a bounded scalar optimization is used to refine the estimate
    of the corner.

    If the refined curvature does not correspond to a positive maximum
    curvature, the smallest regularization parameter in the search grid is
    used as a fallback.

    References
    ----------
    .. [1] P. C. Hansen, *Discrete Inverse Problems: Insight and
           Algorithms*, SIAM, Philadelphia, 2010.

    .. [2] P. C. Hansen, "Analysis of discrete ill-posed problems by means
           of the L-curve," SIAM Review, vol. 34, no. 4, pp. 561--580,
           1992.
    """
    p = len(s)
    beta = np.conj(u.T) @ b # data projection 
    #b0 = b - u @ beta # the least-squares residual contribution in the overdetermined case
    #beta_perp_sq = np.linalg.norm(b) ** 2 - np.linalg.norm(beta)**2
    beta_perp_sq = max(np.linalg.norm(b)**2 - np.linalg.norm(beta)**2, 0.0)
    xi = beta[:p]/s  
    xi[np.isinf(xi)] = 0
    # Filter factors, residual and solution norms
    M, L = u.shape
    eta, rho = f_eta_rho(reg_param, s, xi, beta, beta_perp_sq, M, L)  
    # Call curvature calculator
    curv = np.zeros(len(reg_param))
    for i in np.arange(len(reg_param)):
        curv[i] = curvature(reg_param[i], s, beta, xi) # ok    
    # Initial minimization    
    lower_bound, upper_bound, tolerance = get_bounds(vec_to_minimize = curv, 
                                                     reg_par_vec = reg_param)
    # Refined minimization
    reg_c = optimize.fminbound(curvature, lower_bound, upper_bound, 
                               args = (s, beta, xi), xtol=tolerance,
        full_output=False, disp=False)
    # Final evaluation
    kappa_max = -curvature(reg_c, s, beta, xi) # Maximum curvature.
    if kappa_max < 0:
        lr = len(rho)
        reg_c = reg_param[lr-1]
        rho_c = rho[lr-1]
        eta_c = eta[lr-1]
    else:
        f = np.divide((s**2), (s**2 + reg_c**2))
        eta_c = np.linalg.norm(f * xi)
        rho_c = np.linalg.norm((1-f) * beta[0:len(f)])
        if M > L:
            rho_c = np.sqrt(rho_c ** 2 + beta_perp_sq)
    return reg_c, curv, rho_c, eta_c, eta, rho

def curvature(lambda_val, s, beta, xi):
    """
    Compute the negative curvature of the L-curve [1]_.

    Evaluates the curvature of the L-curve associated with Tikhonov
    regularization for a given regularization parameter :math:`\\lambda`.
    The L-curve is represented in logarithmic coordinates by the residual
    norm

    .. math::

        \\rho(\\lambda)
        =
        \\|\\mathbf{A}\\mathbf{x}_{\\lambda} - \\mathbf{b}\\|_2,

    and the solution norm

    .. math::

        \\eta(\\lambda)
        =
        \\|\\mathbf{x}_{\\lambda}\\|_2.

    Using the singular value decomposition of the forward matrix, the
    Tikhonov filter factors are

    .. math::

        \\varphi_i(\\lambda)
        =
        \\frac{\\sigma_i^2}
        {\\sigma_i^2 + \\lambda^2}.

    The first and second derivatives of :math:`\\rho(\\lambda)` and
    :math:`\\eta(\\lambda)` with respect to :math:`\\lambda` are evaluated
    analytically. These derivatives are then used to compute the curvature
    of the L-curve in the 
    :math:`(\\log(\\rho), \\log(\\eta)) = (\\hat{\\rho}, \\hat{\\eta})` plane.

    The negative curvature is returned so that the L-curve corner can be
    located using a numerical minimization routine [2]_.

    Parameters
    ----------
    lambda_val : float
        Tikhonov regularization parameter :math:`\\lambda`.
    s : ndarray, shape (p,)
        Singular values :math:`\\sigma_i` used in the SVD-based
        regularization.
    beta : ndarray
        Expansion coefficients of the measurement vector in the left
        singular-vector basis,

        .. math::

            \\boldsymbol{\\beta}
            =
            \\mathbf{U}^{H}\\mathbf{b}.

        Depending on the decomposition, ``beta`` may also contain a
        contribution associated with the least-squares residual.
    xi : ndarray, shape (p,)
        SVD coefficients of the unregularized solution, defined by

        .. math::

            \\xi_i = \\frac{\\beta_i}{\\sigma_i}.

    Returns
    -------
    curv : float
        Negative curvature of the L-curve evaluated at ``lambda_val``.

    Notes
    -----
    The curvature is evaluated in logarithmic coordinates,

    .. math::
    
            \\kappa(\\lambda)
            =
            \\frac{
            \\hat{\\rho}^{\\prime}\\hat{\\eta}^{\\prime\\prime}
            -
            \\hat{\\rho}^{\\prime\\prime}\\hat{\\eta}^{\\prime}
            }
            {
            \\left[
            (\\hat{\\rho}^{\\prime})^2
            +
            (\\hat{\\eta}^{\\prime})^2
            \\right]^{3/2}
            },

    where the derivatives are taken with respect to
    :math:`\\lambda`. The function returns :math:`-\\kappa(\\lambda)` to
    allow the maximum-curvature point to be obtained using a minimization
    algorithm.

    The implementation follows the SVD-based formulation of the L-curve
    criterion described by Hansen.

    References
    ----------
    .. [1] P. C. Hansen, *Discrete Inverse Problems: Insight and
        Algorithms*, SIAM, Philadelphia, 2010.

    .. [2] P. C. Hansen, "Analysis of discrete ill-posed problems by means
        of the L-curve," SIAM Review, vol. 34, no. 4, pp. 561--580,
        1992.
    """


    if len(beta) > len(s): # A possible least squares residual.
        LS = True
        rhoLS2 = beta[-1] ** 2
        beta = beta[0:-2]
    else:
        LS = False
    # Filter factors
    f  = np.divide((s ** 2), (s ** 2 + lambda_val ** 2)) 
    #cf = 1 - f # Filter factors complement
    cf = complement_filter_factors(lambda_val, s, check_s_sq = False)
    eta = np.linalg.norm(f * xi) 
    rho = np.linalg.norm(cf * beta)
    f1 = -2 * f * cf / lambda_val 
    f2 = -f1 * (3 - 4*f) / lambda_val
    phi  = np.sum(f*f1*np.abs(xi)**2) 
    psi = np.sum(cf*f1*np.abs(beta)**2)
    dphi = np.sum((f1**2 + f*f2)*np.abs(xi)**2)
    dpsi = np.sum((-f1**2 + cf*f2)*np.abs(beta)**2) 
    # Take care of a possible least squares residual.
    if LS: 
        rho = np.sqrt(rho ** 2 + rhoLS2)
    # First and second derivatives of eta and rho w.r.t lambda;
    deta  =  phi/eta 
    drho  = -psi/rho 
    ddeta =  dphi/eta - deta*deta/eta 
    ddrho = -dpsi/rho - drho*drho/rho 
    # Convert to derivatives of log(eta) and log(rho).
    dlogeta  = deta/eta 
    dlogrho  = drho/rho 
    ddlogeta = ddeta/eta - dlogeta**2 
    ddlogrho = ddrho/rho - dlogrho**2 
    # curvature.
    curv = - np.divide((dlogrho * ddlogeta - ddlogrho * dlogeta),
        (dlogrho**2 + dlogeta**2)**(1.5))
    return curv

def gcv_lambda(u, s, b, r_min = 16 * np.finfo(float).eps, 
               plot_gcvfun = False, plot_in_color = True):
    """
    Select the Tikhonov regularization parameter using generalized
    cross-validation (GCV) [1]_.

    Computes the GCV function over an initial logarithmically spaced grid
    of candidate regularization parameters and refines the minimum using
    a bounded one-dimensional optimization.

    The measurement vector is projected onto the left singular-vector
    basis and the component orthogonal to this subspace is included in
    the GCV residual term. The GCV function itself is evaluated using
    :func:`~pyregtools.regchoice.reg_choice.gcvfun`.

    Parameters
    ----------
    u : ndarray
        Matrix containing the left singular vectors of the forward matrix.
    s : ndarray, shape (p,)
        Singular values, ordered from largest to smallest.
    b : ndarray, shape (M,)
        Measurement vector.
    r_min : float, default=16 * eps
        Minimum regularization parameter relative to the largest singular
        value. The lower bound of the initial search grid is constrained by
        ``s[0] * r_min``, where ``eps`` denotes floating-point machine
        precision.
    plot_gcvfun : bool, default=False
        If ``True``, plot the GCV function together with the selected
        regularization parameter.
    plot_in_color : bool, default=True
        If ``True``, use colored lines and markers in the plot. If ``False``,
        use a grayscale representation. This parameter is only used when
        ``plot_gcvfun=True``.

    Returns
    -------
    reg_min : float
        Tikhonov regularization parameter corresponding to the minimum of
        the GCV function.

    Notes
    -----
    The projected data coefficients are computed as

    .. math::

        \\boldsymbol{\\beta}
        =
        \\mathbf{U}^{H}\\mathbf{b}.

    The squared norm of the component of the measurement vector orthogonal
    to the subspace spanned by the available left singular vectors is
    computed as

    .. math::

        \\beta_{\\perp}^{2}
        =
        \\max\\left(
        \\|\\mathbf{b}\\|_2^2
        -
        \\|\\boldsymbol{\\beta}\\|_2^2,
        0
        \\right),

    where the maximum with zero prevents small negative values caused by
    floating-point roundoff.

    For the compact SVD convention used by PyRegTools, the dimensional
    correction in the GCV denominator is obtained from the shape of
    :math:`\\mathbf{U}` as :math:`M - L`.

    References
    ----------
    .. [1] P. C. Hansen, *Discrete Inverse Problems: Insight and
           Algorithms*, SIAM, Philadelphia, 2010.
    """
    # Sizes and shapes
    npoints = 200  # Number of points in the initial search grid
    M,L = u.shape
    p = len(s)
    beta = np.conj(u).T @ b
    #beta_perp_sq = np.linalg.norm(b) ** 2 - np.linalg.norm(beta)**2
    beta_perp_sq = max(np.linalg.norm(b)**2 - np.linalg.norm(beta)**2, 0.0)
    # Initial search grid
    reg_param = get_regpar(s_valid = s, npoints = npoints, r_min = r_min)
    # Intrinsic residual.
    #delta0 = 0
    #if (m > n and beta_perp_sq > 0):
    #    delta0 = beta_perp_sq
    # Vector of GCV-function values.
    G = np.zeros(npoints)
    for i in np.arange(npoints):
        G[i] = gcvfun(reg_param[i], s, beta[:p], beta_perp_sq, 
                      check_s_sq = False, ml = M-L)
        
    # Initial minimization    
    lower_bound, upper_bound, tolerance = get_bounds(vec_to_minimize = G, 
                                                     reg_par_vec = reg_param)
    # Refined minimization
    reg_min = optimize.fminbound(gcvfun, lower_bound, upper_bound, 
                                args = (s, beta[:p], beta_perp_sq, False,  M-L), 
                                xtol=tolerance, full_output=False, disp=False)
    # Final evalutaion of GCV funtion
    minG = gcvfun(reg_min, s, beta[:p], beta_perp_sq, False, M-L)

    if plot_gcvfun:
        if plot_in_color:
            color_dict = dict(g_c = 'dodgerblue', lam_c = 'r')
        else:
            color_dict = dict(g_c = 'k', lam_c = 'grey')
        
        plt.figure(figsize = (6,3))
        plt.loglog(reg_param , G, linewidth = 1.5, color = color_dict['g_c'])
        plt.loglog(reg_min, minG, marker = 'o', color = color_dict['lam_c'],
                   markerfacecolor = 'none')
        plt.ylim((10**np.floor(np.log10(G.min())), 10**np.ceil(np.log10(G.max()))))
        plt.vlines(x=reg_min, ymin=plt.ylim()[0], ymax=minG, color=color_dict['lam_c'], 
                   linestyle=':', linewidth = 0.7)
        plt.hlines(y=minG, xmin=plt.xlim()[0], xmax=reg_min, color=color_dict['lam_c'], 
                   linestyle=':', linewidth = 0.7)
        plt.xlim((reg_param.min(), reg_param.max()))
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$G(\lambda)$')
        plt.title(r'GCV function ($\lambda = {:.6f})$'.format(reg_min), loc = 'right')
        plt.grid(linestyle = '--')
        plt.tight_layout()    
    return reg_min
        
def gcvfun(lambda_val, s, beta, beta_perp_sq, check_s_sq = False, ml = 0):
    """
    Evaluate the generalized cross-validation (GCV) function.

    Computes the GCV function for a given Tikhonov regularization
    parameter :math:`\\lambda`. The implementation follows the SVD-based
    formulation of the GCV function described by Hansen [1]_.

    For Tikhonov regularization, the usual filter factors are

    .. math::

        \\varphi_i(\\lambda)
        =
        \\frac{\\sigma_i^2}
        {\\sigma_i^2 + \\lambda^2}.

    Since the GCV criterion is based on the residual, the complementary
    filter factors are used 
    (see :func:`~pyregtools.regchoice.reg_choice.complement_filter_factors`),

    .. math::

        \\bar{f}_i(\\lambda)
        =
        1 - \\varphi_i(\\lambda) = \\frac{\\lambda^2}
                {\\sigma_i^2 + \\lambda^2},

    The GCV function is then evaluated as

    .. math::

        G(\\lambda)
        =
        \\frac{
        \\left\\|
        \\bar{\\mathbf{f}}(\\lambda) \\odot \\boldsymbol{\\beta}
        \\right\\|_2^2
        +
        \\beta_2
        }
        {
        \\left[
        M-L+
        \\sum\\limits_i \\bar{f}_i(\\lambda)
        \\right]^2
        },

    where :math:`\\odot` represent an element-wise multiplication and 
    :math:`\\beta = \\mathbf{U}^H\\mathbf{b}` are the coefficients of  
    the measurement vector in the left singular-vector basis; 
    :math:`\\beta_2 = \\|\\mathbf{b}\\|_2^2 - \\|\\mathbf{U}^H\\mathbf{b}\\|_2^2` 
    accounts for a possible residual component outside the subspace represented 
    by the retained singular vectors.

    Parameters
    ----------
    lambda_val : float
        Tikhonov regularization parameter :math:`\\lambda`.
    s : ndarray, shape (p,)
        Singular values.
    beta : ndarray, shape (p,)
        Expansion coefficients of the measurement vector in the left
        singular-vector basis.
    beta_perp_sq : float
        Squared norm of the component of the measurement vector outside
        the column space represented by :math:`\\mathbf{U}`, computed as
        :math:`\\|\\mathbf{b}\\|_2^2 -
        \\|\\mathbf{U}^H\\mathbf{b}\\|_2^2`.
    check_s_sq : bool, default=False
        If ``True``, assumes that both the singular values and the
        regularization parameter are already squared, avoiding repeated
        computation of these quantities when evaluating the filter factors.
    ml : int, default=0
        Dimensional correction in the GCV denominator. For the compact SVD
        used by PyRegTools, this corresponds to :math:`\\text{max}(M - L, 0)`.

    Returns
    -------
    G : float
        Value of the GCV function evaluated at ``lambda_val``.

    References
    ----------
    .. [1] P. C. Hansen, *Discrete Inverse Problems: Insight and
           Algorithms*, SIAM, Philadelphia, 2010.
    """
    cf = complement_filter_factors(lambda_val, s, check_s_sq = check_s_sq)
    G = (np.linalg.norm(cf * beta)**2 + beta_perp_sq)/(ml + np.sum(cf))**2
    return G

def ncp(u, s, b, r_min = 16 * np.finfo(float).eps,
        plotcp = False, plot_in_color = True):
    """
    Select the Tikhonov regularization parameter using the normalized
    cumulative periodogram (NCP) criterion.

    The NCP criterion selects the regularization parameter for which the
    residual most closely resembles white noise [1]_. The criterion is first
    evaluated over a logarithmically spaced grid of candidate
    regularization parameters and the minimum is subsequently refined
    using a bounded one-dimensional optimization.

    The NCP criterion itself is evaluated using
    :func:`~pyregtools.regchoice.reg_choice.ncpfun`.

    Parameters
    ----------
    u : ndarray
        Matrix containing the left singular vectors of the forward matrix.
    s : ndarray, shape (p,)
        Singular values, ordered from largest to smallest.
    b : ndarray, shape (M,)
        Measurement vector.
    r_min : float, default=16 * eps
        Minimum regularization parameter relative to the largest singular
        value. The lower bound of the initial search grid is constrained by
        ``s[0] * r_min``, where ``eps`` denotes floating-point machine
        precision.
    plotcp : bool, default=False
        If ``True``, plot the normalized cumulative periodogram associated
        with the selected regularization parameter together with the
        theoretical white-noise NCP.
    plot_in_color : bool, default=True
        If ``True``, use colored lines in the plot. If ``False``, use a
        grayscale representation. This parameter is only used when
        ``plotcp=True``.

    Returns
    -------
    reg_min_result : float
        Tikhonov regularization parameter selected by the NCP criterion.

    Notes
    -----
    The measurement vector is first projected onto the left singular-vector
    basis,

    .. math::

        \\boldsymbol{\\beta}
        =
        \\mathbf{U}^{H}\\mathbf{b}.

    For each candidate value of :math:`\\lambda`, the normalized cumulative
    periodogram of the regularized residual is compared with the theoretical
    NCP of white noise. The selected regularization parameter minimizes the
    Euclidean distance between these two curves.

    When ``plotcp=True``, a subset of the NCP curves evaluated over the
    initial search grid is shown in gray for reference, together with the
    optimal NCP and the theoretical white-noise NCP.

    References
    ----------
    .. [1] P. C. Hansen, *Discrete Inverse Problems: Insight and
           Algorithms*, SIAM, Philadelphia, 2010.
    """
    # Sizes and shapes
    m = u.shape[0]
    #p = len(s)
    npoints, nNCPs = 200, 20
    # beta - b projection on U
    beta = np.conj(u.T) @ b
    # Initial search grid
    reg_param = get_regpar(s_valid = s, npoints = npoints, r_min = r_min)
    # Inits
    dists = np.zeros(npoints) # Norm of cp-c_white
    q = get_q_ncp(beta, m)
    cp = np.zeros((q-1, npoints)) # cp
    # Fill cp and norm of cp-c_white
    for i in range(npoints):
        dists[i], cp[:, i], _ = ncpfun(reg_param[i], s, beta, u)
    # Initial minimization    
    lower_bound, upper_bound, tolerance = get_bounds(vec_to_minimize = dists, 
                                   reg_par_vec = reg_param)
    # Final miminization
    reg_min_result = optimize.fminbound(clean_ncpfun, lower_bound, upper_bound, 
                                args = (s, beta, u), 
                                xtol=tolerance, full_output=False, disp=False)
    # Final evalutaion of NCP funtion
    dist, cp_opt, cp_white = ncpfun(reg_min_result, s,  beta, u)
    # Print 
    if plotcp:
        if plot_in_color:
            color_dict = dict(cp_opt_c = 'dodgerblue', c_w_c = 'k')
        else:
            color_dict = dict(cp_opt_c = 'k', c_w_c = 'grey')
            
        stp = int(npoints/nNCPs)
        plt.figure(figsize = (6,3))
        plt.plot(cp[:,0:npoints:stp], '-.', color = 'grey', linewidth = 0.5)
        plt.plot(cp_opt, '-', color = color_dict['cp_opt_c'], linewidth = 1.5, 
                 label = r'most white $\mathbf{{c}}(\mathbf{{r}}_{{\lambda}})$')
        plt.plot(cp_white, '--', color = color_dict['c_w_c'], linewidth = 1.5, 
                 label = r'$\mathbf{{c}}_{{\text{white}}}$')
        plt.legend()
        plt.grid(linestyle = '--')
        plt.xlabel('i')
        plt.ylabel(r'$\mathbf{{c}}(\mathbf{{r}}_{{\lambda}})$')
        plt.title(r'$\lambda = {}$'.format(reg_min_result), loc = 'right')
        plt.xlim((0,q-2))
        plt.ylim((-0.1,1.1))
        plt.tight_layout()
    
    return reg_min_result

def ncpfun(lambda_val, s, beta, u, check_s_sq=False):
    """
    Evaluate the normalized cumulative periodogram (NCP) criterion.

    Computes the normalized cumulative periodogram of the Tikhonov
    residual for a given regularization parameter :math:`\\lambda`.
    The optimal regularization parameter is the one for which the
    residual spectrum most closely resembles white noise [1]_.

    The residual is reconstructed from the complementary Tikhonov filter
    factors,

    .. math::

        \\bar{f}_i(\\lambda)
        =
        \\frac{\\lambda^2}
        {\\sigma_i^2 + \\lambda^2},

    and the projected data coefficients,

    .. math::

        \\mathbf{r}(\\lambda)
        =
        \\mathbf{U}
        \\left(
        \\bar{\\mathbf{f}}(\\lambda)
        \\odot
        \\boldsymbol{\\beta}
        \\right).

    The power spectrum of the residual is computed from its discrete
    Fourier transform, and the normalized cumulative periodogram is
    compared with the cumulative periodogram of an ideal white-noise
    sequence.

    Parameters
    ----------
    lambda_val : float
        Tikhonov regularization parameter :math:`\\lambda`.
    s : ndarray, shape (p,)
        Singular values.
    beta : ndarray, shape (p,)
        Expansion coefficients of the measurement vector in the left
        singular-vector basis.
    u : ndarray, shape (M, p)
        Matrix containing the left singular vectors of the forward matrix.
    check_s_sq : bool, default=False
        If ``True``, assumes that both the singular values and the
        regularization parameter are already squared, avoiding repeated
        computation of these quantities when evaluating the filter factors.

    Returns
    -------
    dist : float
        Euclidean distance between the residual normalized cumulative
        periodogram and the ideal white-noise cumulative periodogram.
    cp : ndarray
        Normalized cumulative periodogram of the residual.
    c_white : ndarray
        Ideal normalized cumulative periodogram of white noise.

    Notes
    -----
    The normalized cumulative periodogram is obtained from the residual
    power spectrum,

    .. math::

        d_k(\\lambda)
        =
        \\left|
        \\mathcal{F}\\left\\{
        \\mathbf{r}(\\lambda)
        \\right\\}_k
        \\right|^2.

    where :math:`\\mathcal{F}` denotes the discrete Fourier transform, and 
    :math:`d_k(\\lambda)` is the k-th component of the spectrum of 
    :math:`\\mathbf{r}(\\lambda)`. Thus, the spectrum is a vector 
    :math:`\\mathbf{d} = [d_0, d_1, \\ldots, d_q]^T`. The cumulative periodogram 
    is normalized by the total spectral energy, producing a monotonically 
    increasing curve ranging from 0 to 1, with components given as

    .. math::

        c_k(\\lambda)
        =
        \\frac{
        \\sum\\limits_{i=1}^{k} d_i(\\lambda)
        }
        {
        \\sum\\limits_{i=1}^{q-1} d_i(\\lambda)
        }.

    The white noise theoretical cumulative periodogram is given by

    .. math::

        c_k^{\\mathrm{w}}
        =
        \\frac{k}{q-1},
        \\qquad
        k = 1, \\ldots, q-1.

    The NCP criterion minimizes the distance between this curve and the
    theoretical cumulative periodogram of white noise.

    References
    ----------
    .. [1] P. C. Hansen, *Discrete Inverse Problems: Insight and
           Algorithms*, SIAM, Philadelphia, 2010.
    """
    # Filter factors
    f = complement_filter_factors(lambda_val, s, check_s_sq = check_s_sq)
    # Resudual norm, m and q
    r = u @ (f * beta)
    m = len(r)   
    q = get_q_ncp(beta, m)
    # FFT of residual norm and its NCP
    D = np.abs(np.fft.fft(r)) ** 2
    D = D[1:q]
    cp = np.cumsum(D) / np.sum(D) # NCP of r
    # NCP of white noise
    c_white = np.arange(1, q) / (q-1)
    # distance (norm of cp-v)
    dist = np.linalg.norm(cp - c_white)
    return dist, cp, c_white

def clean_ncpfun(lambda_val, s, beta, u, check_s_sq=False):
    """
    Evaluate the scalar NCP criterion for numerical optimization.

    Wrapper around :func:`ncpfun` that returns only the distance between
    the residual NCP and the theoretical white-noise NCP. This scalar
    output is used by bounded minimization routines.

    Parameters
    ----------
    lambda_val : float
        Tikhonov regularization parameter.
    s : ndarray, shape (p,)
        Singular values.
    beta : ndarray, shape (p,)
        Expansion coefficients of the measurement vector in the left
        singular-vector basis.
    u : ndarray
        Matrix containing the left singular vectors.
    check_s_sq : bool, default=False
        Passed to :func:`ncpfun`.

    Returns
    -------
    dist : float
        NCP distance evaluated at ``lambda_val``.
    """
    dist, _, _ = ncpfun(lambda_val, s, beta, u, check_s_sq = check_s_sq)
    return dist

def get_q_ncp(beta, p):
    """
    Determine the number of Fourier coefficients used by the NCP criterion.

    For real-valued data, only the non-negative frequency components are
    required because the discrete Fourier transform is conjugate symmetric.
    For complex-valued data, the full spectrum is retained.

    Parameters
    ----------
    beta : ndarray
        Expansion coefficients of the measurement vector in the left
        singular-vector basis. Used to determine whether the data are real
        or complex.
    p : int
        Number of samples in the residual vector.

    Returns
    -------
    q : int
        Number of Fourier coefficients considered by the NCP criterion.
        For real-valued data, :math:`q = \\left\\lfloor p/2 \\right\\rfloor + 1` 
        (with :math:`\\left\\lfloor \\cdot \\right\\rfloor` being the floor opperation); 
        for complex-valued data, :math:`q = p`.
    """
    if np.isrealobj(beta):
        q = p // 2 + 1
    else:
        q = p
    return q


def discrep(U, s, V, b, delta, x_0=None):
    m = U.shape[0]
    n = V.shape[0]
    p = len(s)
    ps = 1
    ld = 1
    x_delta = np.zeros((n, ld))
    lambda_val = np.zeros(ld)
    rho = np.zeros(p)
    
    if np.min(delta) < 0:
        raise ValueError("Illegal inequality constraint delta")
    
    if x_0 is None:
        x_0 = np.zeros(n)
    
    if ps == 1:
        omega = np.dot(V.T, x_0)
    else:
        omega = np.linalg.solve(V, x_0)
    
    beta = np.dot(U.T, b)
    delta_0 = np.linalg.norm(b - np.dot(U, beta))
    rho[p - 1] = delta_0 ** 2
    
    if ps == 1:
        for i in range(p - 1, 0, -1):
            rho[i - 1] = rho[i] + (beta[i] - s[i] * omega[i]) ** 2
    else:
        for i in range(0, p - 1):
            rho[i + 1] = rho[i] + (beta[i] - s[i, 0] * omega[i]) ** 2
    
    if np.min(delta) < delta_0:
        raise ValueError("Irrelevant delta < || (I - U*U'')*b ||")
    
    if ps == 1:
        s2 = s ** 2
        for k in range(ld):
            if delta ** 2 >= np.linalg.norm(beta - s * omega) ** 2 + delta_0 ** 2:
                x_delta[:, k] = x_0
            else:
                kmin = np.argmin(np.abs(rho - delta ** 2))
                lambda_0 = s[kmin]
                lambda_val[k] = newton(lambda_0, delta, s, beta, omega, delta_0)
                e = s / (s2 + lambda_val[k] ** 2)
                f = s * e
                x_delta[:, k] = np.dot(V[:, :p], e * beta + (1 - f) * omega)
    elif m >= n:
        omega = omega[:p]
        gamma = s[:, 0] / s[:, 1]
        x_u = np.dot(V[:, p:n], beta[p:n])
        for k in range(ld):
            if delta[k] ** 2 >= np.linalg.norm(beta[:p] - s[:, 0] * omega) ** 2 + delta_0 ** 2:
                x_delta[:, k] = np.dot(V, np.hstack((omega, np.dot(U[:, p:n].T, b))))
            else:
                kmin = np.argmin(np.abs(rho - delta[k] ** 2))
                lambda_0 = gamma[kmin]
                lambda_val[k] = newton(lambda_0, delta[k], s, beta[:p], omega, delta_0)
                e = gamma / (gamma ** 2 + lambda_val[k] ** 2)
                f = gamma * e
                x_delta[:, k] = np.dot(V[:, :p], (e * beta[:p] / s[:, 1]) + (1 - f) * s[:, 1] * omega) + x_u
    else:
        omega = omega[:p]
        gamma = s[:, 0] / s[:, 1]
        x_u = np.dot(V[:, p:m], beta[p:m])
        for k in range(ld):
            if delta[k] ** 2 >= np.linalg.norm(beta[:p] - s[:, 0] * omega) ** 2 + delta_0 ** 2:
                x_delta[:, k] = np.dot(V, np.hstack((omega, np.dot(U[:, p:m].T, b))))
            else:
                kmin = np.argmin(np.abs(rho - delta[k] ** 2))
                lambda_0 = gamma[kmin]
                lambda_val[k] = newton(lambda_0, delta[k], s, beta[:p], omega, delta_0)
                e = gamma / (gamma ** 2 + lambda_val[k] ** 2)
                f = gamma * e
                x_delta[:, k] = np.dot(V[:, :p], (e * beta[:p] / s[:, 1]) + (1 - f) * s[:, 1] * omega) + x_u
    
    return x_delta, lambda_val

def newton(lambda_0, delta, s, beta, omega, delta_0):
    thr = np.sqrt(np.finfo(float).eps)
    it_max = 50
    
    if lambda_0 < 0:
        raise ValueError("Initial guess lambda_0 must be nonnegative")
    
    p = len(s)
    ps = 1
    
    if ps == 2:
        sigma = s[:, 0]
        s = s[:, 0] / s[:, 1]
    
    s2 = s ** 2
    lambda_val = lambda_0
    step = 1
    it = 0
    
    while (abs(step) > thr * lambda_val and abs(step) > thr and it < it_max):
        it += 1
        f = s2 / (s2 + lambda_val ** 2)
        
        if ps == 1:
            r = (1 - f) * (beta - s * omega)
            z = f * r
        else:
            r = (1 - f) * (beta - sigma * omega)
            z = f * r
        
        step = (lambda_val / 4) * (np.dot(r.T, r) + (delta_0 + delta) * (delta_0 - delta)) / np.dot(z.T, r)
        lambda_val -= step
        
        if lambda_val < 0:
            lambda_val = 0.5 * lambda_0
            lambda_0 = 0.5 * lambda_0
    
    if abs(step) > thr * lambda_val and abs(step) > thr:
        raise ValueError("Max. number of iterations ({}) reached".format(it_max))
    
    return lambda_val

def get_regpar_old(s_valid, npoints = 200):
    """
    Deprecated implementation of the L-curve criterion.

    .. deprecated:: 0.1.0
        This function is retained for compatibility and validation purposes.
        Use :func:`get_regpar` instead.
    """
    smin_ratio = 16 * np.finfo(float).eps
    reg_par_grid = np.zeros(npoints)
    reg_par_grid[npoints - 1] = max([s_valid[-1], s_valid[0] * smin_ratio])
    ratio = (s_valid[0] / reg_par_grid[npoints - 1]) ** (1 / (npoints - 1))
    for i in range(npoints - 2, -1, -1):
        reg_par_grid[i] = ratio * reg_par_grid[i + 1]
    return reg_par_grid

def l_curve_old(u, s, b, smin_ratio = 16 * np.finfo(float).eps, 
            plotit = False, plot_in_color = True):
    """
    Deprecated implementation of the L-curve criterion.

    .. deprecated:: 0.1.0
        This function is retained for compatibility and validation purposes.
        Use :func:`l_curve` instead.
    """
    # Sizes and shapes
    npoints = 200  # Number of points on the L-curve
    # smin_ratio = 16*np.finfo(float).eps  # Smallest regularization parameter.
    Nm, Nu = u.shape
    p = s.shape#len(s)
    beta = np.conjugate(u).T @ b
    beta_perp_sq = np.linalg.norm(b) ** 2 - np.linalg.norm(beta)**2
    beta = np.reshape(beta[0:int(p[0])], beta.shape[0])
    xi = np.divide(beta[0:int(p[0])],s)
    # beta = np.reshape(beta[:p], beta.shape[0])
    # xi = np.divide(beta[:p], s)
    xi[np.isinf(xi)] = 0
    # Initial search grid
    # reg_param = get_regpar(s_valid = s[:p], npoints = npoints, smin_ratio = smin_ratio)


    eta = np.zeros((npoints,1))
    rho = np.zeros((npoints,1)) #eta
    reg_param = np.zeros((npoints,1))
    s2 = s ** 2
    reg_param[-1] = np.amax([s[-1], s[0]*smin_ratio])
    ratio = (s[0]/reg_param[-1]) ** (1/(npoints-1))
    for i in np.arange(start=npoints-2, step=-1, stop = -1):
        reg_param[i] = ratio*reg_param[i+1]
    for i in np.arange(start=0, step=1, stop = npoints):
        f = s2 / (s2 + reg_param[i] ** 2) # filter factors
        eta[i] = np.linalg.norm(f * xi) # solution norm
        rho[i] = np.linalg.norm((1-f) * beta[:int(p[0])]) # residual norm
    if (Nm > Nu and beta_perp_sq > 0):
        rho = np.sqrt(rho ** 2 + beta_perp_sq)
    # Compute the corner of the L-curve (optimal regularization parameter)
    lam_opt, reg_param, curv, rho_c, eta_c = l_corner_old(rho,eta,reg_param,u,s,b)
    lam_opt = lam_opt[0]
    # want to plot the L curve?
    if plotit:
        if plot_in_color:
            color_dict = dict(l_c = 'dodgerblue', lam_l = 'r', c_c = 'navy')
        else:
            color_dict = dict(l_c = 'k', lam_l = 'grey', c_c = 'k')
        fig = plt.figure(figsize = (6,3))
        # L-curve
        plt.loglog(rho, eta, color = color_dict['l_c'], linewidth = 1.5)
        plt.loglog(rho_c, eta_c, marker = 'o', color = color_dict['lam_l'],
                   markerfacecolor = 'none')
        plt.xlim((10**np.floor(np.log10(rho.min())), 10**np.ceil(np.log10(rho.max()))))
        plt.ylim((10**np.floor(np.log10(eta.min())), 10**np.ceil(np.log10(eta.max()))))

        plt.vlines(x = rho_c, ymin=plt.ylim()[0], ymax = eta_c, color=color_dict['lam_l'], 
                   linestyle=':', linewidth = 0.7)
        plt.hlines(y = eta_c, xmin=plt.xlim()[0], xmax = rho_c, color=color_dict['lam_l'], 
                   linestyle=':', linewidth = 0.7)
        plt.title(r'L-curve ($\lambda = ${:.6f})'.format(lam_opt), loc = 'right')
        plt.xlabel(r'Residual norm $||Ax - b||_2$')
        plt.ylabel(r'Solution norm $||x||_2$')
        plt.grid(linestyle = '--', which='both')
        plt.tight_layout()
        # Curvature
        ax2 = fig.add_axes([0.60, 0.55, 0.3, 0.3])
        ax2.semilogx(reg_param, -curv, color = color_dict['c_c'], linewidth = 1.5)
        ax2.semilogx(lam_opt, np.amax(-curv), marker = 'o', color = color_dict['lam_l'],
                   markerfacecolor = 'none')
        ax2.set_xlim((reg_param.min(), reg_param.max()))
        ax2.set_ylim((-0.1*(1.2*np.amax(-curv)), 1.2*np.amax(-curv)))
        ax2.vlines(x = lam_opt, ymin=ax2.set_ylim()[0], ymax = np.amax(-curv), 
                   color=color_dict['lam_l'], linestyle=':', linewidth = 0.7)
        ax2.hlines(y = np.amax(-curv), xmin=ax2.set_xlim()[0], xmax = lam_opt, 
                   color=color_dict['lam_l'], linestyle=':', linewidth = 0.7)
        ax2.grid(linestyle = '--')
        ax2.set_xlabel(r'$\lambda$')
        ax2.set_ylabel(r'$-c(\lambda)$')
        plt.show()
    return lam_opt

def l_corner_old(rho,eta,reg_param,u,sig,bm):
    """
    Deprecated implementation of the L-curve criterion.

    .. deprecated:: 0.1.0
        This function is retained for compatibility and validation purposes.
        Use :func:`l_corner` instead.
    """
    # Set threshold for skipping very small singular values in the analysis of a discrete L-curve.
    s_thr = np.finfo(float).eps # Neglect singular values less than s_thr.
    # Set default parameters for treatment of discrete L-curve.
    deg   = 2  # Degree of local smooting polynomial.
    q     = 2  # Half-width of local smoothing interval.
    order = 4  # Order of fitting 2-D spline curve.
    # Initialization.
    if (len(rho) < order):
        print('I will fail. Too few data points for L-curve analysis')
    Nm, Nu = u.shape
    p = sig.shape
    beta = (np.conj(u).T) @ bm 
    beta = np.reshape(beta[0:int(p[0])], beta.shape[0])
    # b0 = (bm - (beta.T @ u).T)
    b0 = bm - u @ beta
    xi = np.divide(beta[0:int(p[0])], sig)
    # Call curvature calculator
    curv = curvature_old(reg_param, sig, beta, xi) # ok
    
    # Minimize 1
    curv_id = np.argmin(curv)
    x1 = reg_param[int(np.amin([curv_id+1, len(curv)-1]))]
    x2 = reg_param[int(np.amax([curv_id-1, 0]))]
    # print(x1)
    # print(x1.shape)
    # x1 = reg_param[int(np.amin([curv_id+1, len(curv)]))]
    # x2 = reg_param[int(np.amax([curv_id-1, 0]))]
    # Minimize 2 - set tolerance first (new versions of scipy need that)
    tolerance_array = np.zeros(len(x1)+len(x2)+1)
    tolerance_array[0:len(x1)] = x1.flatten()
    tolerance_array[len(x1):len(x1)+len(x2)] = x2.flatten()
    tolerance_array[-1] = 1e-5
    # print(tolerance_array)
    tolerance = np.amin(tolerance_array)#np.amin([x1/50, x2/50, 1e-5])
    reg_c = optimize.fminbound(curvature_old, x1, x2, args = (sig, beta, xi), xtol=tolerance,
        full_output=False, disp=False)
    kappa_max = - curvature_old(reg_c, sig, beta, xi) # Maximum curvature.
    if kappa_max < 0:
        lr = len(rho)
        reg_c = reg_param[lr-1]
        rho_c = rho[lr-1]
        eta_c = eta[lr-1]
    else:
        f = np.divide((sig**2), (sig**2 + reg_c**2))
        eta_c = np.linalg.norm(f * xi)
        rho_c = np.linalg.norm((1-f) * beta[0:len(f)])
        if Nm > Nu:
            rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0)**2)
    return reg_c, reg_param, curv, rho_c, eta_c



def curvature_old(lambd, sig, beta, xi):
    """
    Deprecated implementation of the L-curve criterion.

    .. deprecated:: 0.1.0
        This function is retained for compatibility and validation purposes.
        Use :func:`curvature` instead.
    """
    # Initialization.
    phi = np.zeros(lambd.shape)
    dphi = np.zeros(lambd.shape)
    psi = np.zeros(lambd.shape)
    dpsi = np.zeros(lambd.shape)
    eta = np.zeros(lambd.shape)
    rho = np.zeros(lambd.shape)
    if len(beta) > len(sig): # A possible least squares residual.
        LS = True
        rhoLS2 = beta[-1] ** 2
        beta = beta[0:-2]
    else:
        LS = False
    # Compute some intermediate quantities.
    for jl, lam in enumerate(lambd):
        f  = np.divide((sig ** 2), (sig ** 2 + lam ** 2)) # ok
        cf = 1 - f # ok
        eta[jl] = np.linalg.norm(f * xi) # ok
        rho[jl] = np.linalg.norm(cf * beta)
        f1 = -2 * f * cf / lam 
        f2 = -f1 * (3 - 4*f)/lam
        phi[jl]  = np.sum(f*f1*np.abs(xi)**2) #ok
        psi[jl] = np.sum(cf*f1*np.abs(beta)**2)
        dphi[jl] = np.sum((f1**2 + f*f2)*np.abs(xi)**2)
        dpsi[jl] = np.sum((-f1**2 + cf*f2)*np.abs(beta)**2) #ok

    if LS: # Take care of a possible least squares residual.
        rho = np.sqrt(rho ** 2 + rhoLS2)

    # Now compute the first and second derivatives of eta and rho
    # with respect to lambda;
    deta  =  np.divide(phi, eta) #ok
    drho  = -np.divide(psi, rho)
    ddeta =  np.divide(dphi, eta) - deta * np.divide(deta, eta)
    ddrho = -np.divide(dpsi, rho) - drho * np.divide(drho, rho)

    # Convert to derivatives of log(eta) and log(rho).
    dlogeta  = np.divide(deta, eta)
    dlogrho  = np.divide(drho, rho)
    ddlogeta = np.divide(ddeta, eta) - (dlogeta)**2
    ddlogrho = np.divide(ddrho, rho) - (dlogrho)**2
    # curvature.
    curv = - np.divide((dlogrho * ddlogeta - ddlogrho * dlogeta),
        (dlogrho**2 + dlogeta**2)**(1.5))
    
    return curv