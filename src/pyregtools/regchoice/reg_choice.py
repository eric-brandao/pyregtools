""" L-curve regularization parameter finding.

This set of functions are used to find an optimal regularization parameter
for a under determined system of equations. The optimal parameter is found
according to the L-curve criteria, described in

Discrete Inverse Problems - insight and algorithms, Per Christian Hansen,
Technical University of Denmark, DTU compute, 2010

The code was adapted from http://www.imm.dtu.dk/~pcha/Regutools/ by
Per Christian Hansen, DTU Compute, October 27, 2010 - originally implemented in Matlab

This version is only for the under determined case.
"""


import numpy as np
import matplotlib.pyplot as plt
#import scipy.io as scio
#from scipy import linalg # for svd
#from sklearn.linear_model import Ridge
from scipy import optimize
import warnings

def csvd(A):
    """ Computes the SVD based on the size of A.

    Parameters
    ----------
        A : numpy ndarray
            sensing matrix (Nm x Nu). Nm are the number of measurements
            and Nu the number of unknowns
    Returns
    -------
        u : numpy ndarray
            left singular vectors
        sig : numpy 1darray
            singular values
        v : numpy ndarray
            right singular vectors
    """
    Nm, Nu = A.shape
    if Nm >= Nu: # more measurements than unknowns
        u, sig, v = np.linalg.svd(A, full_matrices=False)
        v = np.conjugate(v.T)
    else:
        v, sig, u = np.linalg.svd(np.conjugate(A.T), full_matrices=False)
        u = np.conjugate(u.T)
    return u, sig, v

def gram_matrix(A):
    """ Computes Gram matrix of A
    
    Allows one to analyze if A has many correlated columns or not. Valid for
    complex matrices. Scales from 0 to 1.
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

def get_regpar(s_valid, npoints = 200, smin_ratio = 16 * np.finfo(float).eps):
    """ Get the initial search grid for the regularization parameter
    
    Vectorized version
    
    Parameters
    -----------
    s_valid : numpy1dArray
        valid range of the singular values - usually up to p = len(s)
    npoints : int
        number of points on the grid
    smin_ratio : float
        ratio to first singular value. Odds are that the minimum regularization 
        parameter will be s[0]*smin_ratio
    """
    last_val = max(s_valid[-1], s_valid[0] * smin_ratio)
    reg_par_grid = np.geomspace(s_valid[0], last_val, npoints)
    return reg_par_grid#[reg_par_grid > 1e-6]

def get_regpar_old(s_valid, npoints = 200):
    """ Get the initial search grid for the regularization parameter
    
    Parameters
    -----------
    s_valid : numpy1dArray
        valid range of the singular values - usually up to p = len(s)
    npoints : int
        number of points on the grid
    """
    smin_ratio = 16 * np.finfo(float).eps
    reg_par_grid = np.zeros(npoints)
    reg_par_grid[npoints - 1] = max([s_valid[-1], s_valid[0] * smin_ratio])
    ratio = (s_valid[0] / reg_par_grid[npoints - 1]) ** (1 / (npoints - 1))
    for i in range(npoints - 2, -1, -1):
        reg_par_grid[i] = ratio * reg_par_grid[i + 1]
    return reg_par_grid

def f_eta_rho(reg_param, s, xi, beta, beta2, Nm, Nu):
    """ Compute filter factors, solution and residual norms
    """
    eta = np.zeros(len(reg_param))
    rho = np.zeros(len(reg_param))
    for i, lam in enumerate(reg_param):
        f = s**2 / (s**2 + lam ** 2) # filter factors
        eta[i] = np.linalg.norm(f * xi) # solution norm
        rho[i] = np.linalg.norm((1-f) * beta) # residual norm
    if (Nm > Nu and beta2 > 0):
        rho = np.sqrt(rho ** 2 + beta2)
    return eta, rho

def complement_filter_factors(lambda_val, s, check_s_sq = False):
    """ Compute the term (1-filter factors)
    
    Parammeters
    -------------
    lambda_val : float
        Value of the regularization parameter
    s : numpy1dArray
        Singular values
    """
    if not check_s_sq:
        f = (lambda_val ** 2) / (s ** 2 + lambda_val ** 2)
    else:
        f = lambda_val / (s + lambda_val)
    return f

def get_bounds(vec_to_minimize, reg_par_vec):
    """ Get bounds and tolerance for optmization routine
    
    Parameters
    -----------
    vec_to_minimize : numpy1dArray
        vector of function to be minimized
    reg_par_vec : numpy1dArray
        vector of the regularization parameters (initial grid). Bounds for the
        regularization parameter are extracted from this value.
    """
    npoints = len(reg_par_vec)
    min_id = np.argmin(vec_to_minimize)
    # min_g = np.amin(dists)
    # min_g_id = np.where(dists == min_g)[0][0]
    x1 = reg_par_vec[int(np.amin(np.array([min_id+1, npoints-1])))]
    x2 = reg_par_vec[int(np.amax([min_id - 1, 0]))]
    tolerance = np.amin([x1/50, x2/50, 1e-5])
    return x1, x2, tolerance    

def curvature_new(lambda_val, s, beta, xi):
    """ Computes the negative of the curvature of the L-curve.
    
    (see Sec. 5.4 of Discrete Inverse Problems)

    Parameters
    ----------
        lambda_val : float
            regularization parameter
        sig : numpy 1darray
            singular values
        beta: numpy 1darray
            conj(u) @ bm
        xi : numpy 1darray
            beta / sig
    Returns
    -------
        curv : numpy 1darray
            negative curvature
    """
    if len(beta) > len(s): # A possible least squares residual.
        LS = True
        rhoLS2 = beta[-1] ** 2
        beta = beta[0:-2]
    else:
        LS = False
    # Filter factors
    f  = np.divide((s ** 2), (s ** 2 + lambda_val ** 2)) # ok
    cf = 1 - f # Filter factors complement
    eta = np.linalg.norm(f * xi) # ok
    rho = np.linalg.norm(cf * beta)
    f1 = -2 * f * cf / lambda_val 
    f2 = -f1 * (3 - 4*f) / lambda_val
    phi  = np.sum(f*f1*np.abs(xi)**2) #ok
    psi = np.sum(cf*f1*np.abs(beta)**2)
    dphi = np.sum((f1**2 + f*f2)*np.abs(xi)**2)
    dpsi = np.sum((-f1**2 + cf*f2)*np.abs(beta)**2) #ok 
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

def l_corner_new(reg_param,u,s,b):
    """ Computes the corner of the L-curve.

    Uses the function "curvature1d" (see Sec. 5.4 of Discrete Inverse Problems)

    Parameters
    ----------
        rho : numpy 1darray
            computed in l_curve function (residual norm) - related to curvature
        eta : numpy 1darray
            computed in l_curve function (solution norm) - related to curvature
        reg_param : numpy 1darray
            computed in l_curve function
        u : numpy ndarray
            left singular vectors
        sig : numpy 1darray
            singular values
        bm: numpy 1darray
            your measurement vector (size: Nm x 1)
    Returns
    -------
        reg_c : float
            optimal regularization parameter
    """
    p = len(s)
    beta = np.conj(u.T) @ b # data projection 
    b0 = b - u @ beta # residual numerical correction
    beta2 = np.linalg.norm(b) ** 2 - np.linalg.norm(beta)**2
    # U^H @ b/sig
    xi = beta[:p]/s  
    xi[np.isinf(xi)] = 0
    # Filter factors, residual and solution norms
    Nm, Nu = u.shape
    eta, rho = f_eta_rho(reg_param, s, xi, beta, beta2, Nm, Nu)  
    order = 4  # Order of fitting 2-D spline curve.
    # Initialization.
    if (len(rho) < order):
        print('I will fail. Too few data points for L-curve analysis')
    # Call curvature calculator
    curv = np.zeros(len(reg_param))
    for i in np.arange(len(reg_param)):
        curv[i] = curvature_new(reg_param[i], s, beta, xi) # ok    
    # Initial minimization    
    x1, x2, tolerance = get_bounds(vec_to_minimize = curv, reg_par_vec = reg_param)
    # Refined minimization
    reg_c = optimize.fminbound(curvature_new, x1, x2, args = (s, beta, xi), xtol=tolerance,
        full_output=False, disp=False)
    # Final evaluation
    kappa_max = -curvature_new(reg_c, s, beta, xi) # Maximum curvature.
    if kappa_max < 0:
        lr = len(rho)
        reg_c = reg_param[lr-1]
        rho_c = rho[lr-1]
        eta_c = eta[lr-1]
    else:
        f = np.divide((s**2), (s**2 + reg_c**2))
        eta_c = np.linalg.norm(f * xi)
        rho_c = np.linalg.norm((1-f) * beta[0:len(f)])
        if Nm > Nu:
            rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0)**2)
    return reg_c, curv, rho_c, eta_c, eta, rho

def l_curve_new(u, s, b, smin_ratio = 16 * np.finfo(float).eps, 
            plotit = False, plot_in_color = True):
    """ Optimal regularization parameter via the L-curve criterion.

    This function uses the L-curve and computes its curvature in
    order to find its corner - optimal regularization parameter.

    Uses the function "l_corner"

    Parameters
    ----------
        U : numpyndArray
            Left singular vectors matrix
        s : numpy1dArray
            Singular values
        b : numpy1dArray
            Measurement vector
        smin_ratio : float
            ratio to first singular value. Odds are that the minimum regularization 
            parameter will be s[0]*smin_ratio
        plotit : bool
            whether to plot the L curve or not. Default is False
        plot_in_color : bool
            choose to plot the regularization function in color or black and white.
        
    Returns
    -------
        lam_opt : float
            optimal regularization parameter
    """
    # Sizes and shapes
    npoints = 200  # Number of points on the L-curve    
    p = len(s)
    # Initial search grid
    reg_param = get_regpar(s_valid = s[:p], npoints = npoints, smin_ratio = smin_ratio)
    # Compute everything
    lam_opt, curv, rho_c, eta_c, eta, rho = l_corner_new(reg_param,u,s,b)
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

def curvature(lambd, sig, beta, xi):
    """ computes the NEGATIVE of the curvature.

    Parameters
    ----------
        lambd : float
            regularization parameter
        sig : numpy 1darray
            singular values
        beta: numpy 1darray
            conj(u) @ bm
        xi : numpy 1darray
            beta / sig
    Returns
    -------
        curv : numpy 1darray
            negative curvature
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

def l_corner(rho,eta,reg_param,u,sig,bm):
    """ Computes the corner of the L-curve.

    Uses the function "curvature"

    Parameters
    ----------
        rho : numpy 1darray
            computed in l_curve function (residual norm) - related to curvature
        eta : numpy 1darray
            computed in l_curve function (solution norm) - related to curvature
        reg_param : numpy 1darray
            computed in l_curve function
        u : numpy ndarray
            left singular vectors
        sig : numpy 1darray
            singular values
        bm: numpy 1darray
            your measurement vector (size: Nm x 1)
    Returns
    -------
        reg_c : float
            optimal regularization parameter
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
    curv = curvature(reg_param, sig, beta, xi) # ok
    
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
    reg_c = optimize.fminbound(curvature, x1, x2, args = (sig, beta, xi), xtol=tolerance,
        full_output=False, disp=False)
    kappa_max = - curvature(reg_c, sig, beta, xi) # Maximum curvature.
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

def l_curve(u, s, b, smin_ratio = 16 * np.finfo(float).eps, 
            plotit = False, plot_in_color = True):
    """ Optimal regularization parameter via the L-curve criterion.

    This function uses the L-curve and computes its curvature in
    order to find its corner - optimal regularization parameter.

    Uses the function "l_corner"

    Parameters
    ----------
        U : numpyndArray
            Left singular vectors matrix
        s : numpy1dArray
            Singular values
        b : numpy1dArray
            Measurement vector
        smin_ratio : float
            ratio to first singular value. Odds are that the minimum regularization 
            parameter will be s[0]*smin_ratio
        plotit : bool
            whether to plot the L curve or not. Default is False
        plot_in_color : bool
            choose to plot the regularization function in color or black and white.
        
    Returns
    -------
        lam_opt : float
            optimal regularization parameter
    """
    # Sizes and shapes
    npoints = 200  # Number of points on the L-curve
    # smin_ratio = 16*np.finfo(float).eps  # Smallest regularization parameter.
    Nm, Nu = u.shape
    p = s.shape#len(s)
    beta = np.conjugate(u).T @ b
    beta2 = np.linalg.norm(b) ** 2 - np.linalg.norm(beta)**2
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
    if (Nm > Nu and beta2 > 0):
        rho = np.sqrt(rho ** 2 + beta2)
    # Compute the corner of the L-curve (optimal regularization parameter)
    lam_opt, reg_param, curv, rho_c, eta_c = l_corner(rho,eta,reg_param,u,s,b)
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

def gcv_lambda(u, s, b, smin_ratio = 16 * np.finfo(float).eps, 
               plot_gcvfun = False, plot_in_color = True):
    """ Optimal regularization parameter via Generalized Cross Validation.
    
    Finds the optmimal regularization parameter for Tikhonov regularization 
    according to the GCV criterion (see Sec. 5.4 of Discrete Inverse Problems)
    
    Parameters
    ------------
    U : numpyndArray
        Left singular vectors matrix
    s : numpy1dArray
        Singular values
    b : numpy1dArray
        Measurement vector
    smin_ratio : float
        ratio to first singular value. Odds are that the minimum regularization 
        parameter will be s[0]*smin_ratio
    plotcp : bool
        choose to plot the regularization function or not.
    plot_in_color : bool
        choose to plot the regularization function in color or black and white.
    
    Returns
    -------
        lambda : float
            estimated regularization parameter
    """
    # Sizes and shapes
    npoints = 200  # Number of points on the L-curve
    m, n = u.shape
    p = len(s)
    beta = np.conjugate(u).T @ b
    beta2 = np.linalg.norm(b) ** 2 - np.linalg.norm(beta)**2
    # Initial search grid
    reg_param = get_regpar(s_valid = s[:p], npoints = npoints, smin_ratio = smin_ratio)
    # npoints = len(reg_param)
    # Intrinsic residual.
    delta0 = 0
    if (m > n and beta2 > 0):
        delta0 = beta2
    # Vector of GCV-function values.
    G = np.zeros(npoints)
    for i in np.arange(npoints):
        G[i] = gcvfun(reg_param[i], s, beta[:p], delta0, dsvd = False, mn = m-n)
        
    # Initial minimization    
    x1, x2, tolerance = get_bounds(vec_to_minimize = G, reg_par_vec = reg_param)
    # Refined minimization
    reg_min = optimize.fminbound(gcvfun, x1, x2, 
                                args = (s, beta[:p], delta0, False,  m-n), 
                                xtol=tolerance, full_output=False, disp=False)
    # Final evalutaion of GCV funtion
    minG = gcvfun(reg_min, s, beta[:p], delta0, False, m-n)

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
        
def gcvfun(lambda_val, s, beta, delta0, dsvd = False, mn = 0):
    """ GCV function
    
    Parammeters
    -------------
    lambda_val : float
        Value of the regularization parameter
    s : numpy1dArray
        Singular values
    dsvd : bool
        Computational mode of filter factors
    """
    f = complement_filter_factors(lambda_val, s, check_s_sq = dsvd)
    G = (np.linalg.norm(f * beta)**2 + delta0)/(mn + np.sum(f))**2
    return G

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

def get_q_ncp(beta, m):
    """ Get the value of q for NCP
    """
    if np.isrealobj(beta):
        q = m // 2 +1
    else:
        q = m
    return q

def ncp(U, s, b, smin_ratio = 16 * np.finfo(float).eps,
        plotcp = False, plot_in_color = True):
    """ Normalized Cumulative Periodgram regularization criterion
    
    Finds the optmimal regularization parameter for Tikhonov regularization 
    according to the NCP criterion (see Sec. 5.5 of Discrete Inverse Problems)
    
    Parameters
    ------------
    U : numpyndArray
        Left singular vectors matrix
    s : numpy1dArray
        Singular values
    b : numpy1dArray
        Measurement vector
    smin_ratio : float
        ratio to first singular value. Odds are that the minimum regularization 
        parameter will be s[0]*smin_ratio
    plotcp : bool
        choose to plot the regularization function or not.
    plot_in_color : bool
        choose to plot the regularization function in color or black and white.
    """
    # Sizes and shapes
    m = U.shape[0]
    p = len(s)
    npoints, nNCPs = 200, 20
    # beta - b projection on U
    beta = np.conj(U.T) @ b
    # Initial search grid
    reg_param = get_regpar(s_valid = s[:p], npoints = npoints, smin_ratio = smin_ratio)
    # Inits
    dists = np.zeros(npoints) # Norm of cp-c_white
    q = get_q_ncp(beta, m)
    cp = np.zeros((q-1, npoints)) # cp
    # Fill cp and norm of cp-c_white
    for i in range(npoints):
        dists[i], cp[:, i], _ = ncpfun(reg_param[i], s, beta[:p], U[:, :p])
    # Initial minimization    
    x1, x2, tolerance = get_bounds(vec_to_minimize = dists, 
                                   reg_par_vec = reg_param)
    # Final miminization
    reg_min_result = optimize.fminbound(clean_ncpfun, x1, x2, 
                                args = (s[:p], beta[:p], U[:, :p]), 
                                xtol=tolerance, full_output=False, disp=False)
    # Final evalutaion of NCP funtion
    dist, cp_opt, cp_white = ncpfun(reg_min_result, s[:p],  beta[:p], U[:, :p])
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

def ncpfun(lambda_val, s, beta, U, dsvd=False):
    """ NCP function
    
    Parammeters
    -------------
    lambda_val : float
        Value of the regularization parameter
    s : numpy1dArray
        Singular values
    dsvd : bool
        Computational mode of filter factors
    """
    # Filter factors
    f = complement_filter_factors(lambda_val, s, check_s_sq = dsvd)
    # Resudual norm, m and q
    r = U @ (f * beta)
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

def clean_ncpfun(lambda_val, s, beta, U, dsvd=False):
    """ Version of the ncpfun for fminbound optmizer.
    """
    dist, _, _ = ncpfun(lambda_val, s, beta, U, dsvd=False)
    return dist 