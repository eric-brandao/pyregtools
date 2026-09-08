""" Iterative solvers for inverse problems.

This set of functions are used to find regularized solutions using iterative solvers.

Discrete Inverse Problems - insight and algorithms, Per Christian Hansen,
Technical University of Denmark, DTU compute, 2010

The code was adapted from http://www.imm.dtu.dk/~pcha/Regutools/ by
Per Christian Hansen, DTU Compute, October 27, 2010 - originally implemented in Matlab
"""

import numpy as np
import matplotlib.pyplot as plt

class IterativeSolvers(object):
    """ Solve regularized problems with iterative solvers
    """
    def __init__(self, A, b, x0 = None):
        self.A = A
        self.m = self.A.shape[0]
        self.n = self.A.shape[1]
        self.b = b
        if x0 is None:
            self.x0 = np.zeros(self.A.shape[1], dtype = complex)
        else:
            self.x0 = x0
            
    def init_vars(self, ):
        self.xk = np.copy(self.x0)
        self.x_sol = np.zeros((self.n, self.max_it), dtype = complex)
        self.res_norms = np.zeros(self.max_it)
        self.sol_norms = np.zeros(self.max_it)
            
    def landweber_cimmino(self, omega = 0.1, max_it = 50, cimmino = False):
        """ Landweber iterative solver
        
        As in Eq. (6.1) of Discrete Inverse Problems
        """
        self.max_it = max_it
        self.omega = omega
        Ah = np.conj(self.A.T) # Hermitian of A
        AhA_norm = np.linalg.norm(Ah @ self.A)
        # Check omega
        if self.omega<=0 or self.omega>=2/AhA_norm:
            raise ValueError(r"$\omega$ must be a value between 0 and {}".format(2/AhA_norm))   
        # initialize variables
        self.init_vars()
        if cimmino:
            D = self.get_Dmtx()
            Ah = Ah @ np.diag(D)
        # loop
        for k in range(self.max_it):
            rk = self.b - self.A @ self.xk
            self.xk += self.omega * Ah @ rk
            self.x_sol[:,k] = self.xk
            self.res_norms[k] = np.linalg.norm(rk)
            self.sol_norms[k] = np.linalg.norm(self.xk)
            
    def get_Dmtx(self):
        """ Get D matrix for Cimmmino iterative solver
        """
        # matrix D
        D = np.zeros(self.m)
        for row in range(self.m):
            if not np.any(self.A[row,:]):
                D[row] = 0
            else:
                row_norm = np.linalg.norm(self.A[row,:])
                D[row] = 1/(self.m*row_norm**2)
        return D
    
    def art_solver(self, max_it = 50):
        """ Kaczmarz’s method or Algebraic Reconstruction Technique (ART)
        
        As in Sec 6.1.2 of Discrete Inverse Problems
        """
        self.max_it = max_it
        # initialize variables
        self.init_vars()
        # loop through iterations
        for k in range(self.max_it):
            # loop through rows of A
            for i in range(self.m):
                a_i = self.A[i,:]
                scale = (self.b[i]-np.conj(a_i) @ self.xk)/(np.linalg.norm(a_i)**2)
                self.xk += scale * a_i
            self.x_sol[:,k] = self.xk
            self.res_norms[k] = np.linalg.norm(self.A @ self.xk - self.b)
            self.sol_norms[k] = np.linalg.norm(self.xk)
        
    def cgls(self, max_it = 50):
        """ Conjugate Gradient Least-Squares
        
        As in Sec 6.3.2 of Discrete Inverse Problems
        """
        self.max_it = max_it
        Ah = np.conj(self.A.T) # Hermitian of A
        # initialize solution vector
        # initialize variables
        self.init_vars()
        # loop
        rk = self.b - self.A @ self.xk
        dk = Ah @ rk
        normr2 = np.linalg.norm(dk)**2
        for k in range(max_it):
            # past_residual = residual
            # d_k = Ah @ residual 
            Ad = self.A @ dk
            alphak = normr2/(np.linalg.norm(Ad)**2)
            self.xk += alphak * dk
            rk -= alphak * Ad #A @ dk
            sk = Ah @ rk
            normr2_new = np.linalg.norm(sk)**2
            betak = normr2_new/normr2
            normr2 = normr2_new
            dk = sk + betak * dk
            # fill returns
            self.x_sol[:,k] = self.xk
            self.res_norms[k] = np.linalg.norm(rk)
            self.sol_norms[k] = np.linalg.norm(self.xk)

    def plot_l(self, xlim = None, ylim = None):
        """ plots L-curve along iterations
        """
        if xlim is None:
            xlim = (0.9*self.res_norms.min(), 1.1*self.res_norms.max())
        if ylim is None:
            ylim = (0.9*self.sol_norms.min(), 1.1*self.sol_norms.max())
        
        plt.figure(figsize = (6,3))
        plt.loglog(self.res_norms, self.sol_norms, 'o-k', alpha = 0.8, markersize = 3)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(linestyle = '--')
        plt.xlabel(r"$\left\|\mathbf{Ax}-\mathbf{b}\right\|$")
        plt.ylabel(r"$\left\|\mathbf{x}\right\|$")

        plt.tight_layout()
        
    def plot_error_history(self, x_true):
        """ plots error history vs. iterations
        """
        error = self.x_sol - np.reshape(np.repeat(x_true, self.max_it), (self.n, self.max_it))
        e = np.linalg.norm(error, axis = 0)
        plt.figure(figsize = (6,3))
        plt.semilogy(e, 'o-k', markersize = 3)
        plt.grid(linestyle = '--')
        plt.xlabel(r"Iterations $k$ [-]")
        plt.ylabel(r"$\left\|\mathbf{x}_{\text{True}} - \mathbf{x}_{\text{sol}}\right\|$")
        plt.tight_layout()

def landweber(A, b, omega = 0.1, x0 = None, max_it = 50):
    """ Landweber iterative solver
    
    As in Eq. (6.1) of Discrete Inverse Problems
    """
    if x0 is None:
        x0 = np.zeros(A.shape[1], dtype = complex)
    Ah = np.conj(A.T) # Hermitian of A
    AhA_norm = np.linalg.norm(Ah@A)
    # Check omega
    if omega<=0 or omega>=2/AhA_norm:
        raise ValueError(r"$\omega$ must be a value between 0 and {}".format(2/AhA_norm))   
    # initialize solution vector
    x_sol = np.zeros((A.shape[1], max_it), dtype = complex)
    xk = np.copy(x0)
    # loop
    for k in range(max_it):
        residual = b - A @ xk
        xk += omega * Ah @ residual
        x_sol[:,k] = xk
    return x_sol

def landweber_cimmino(A, b, omega = 0.1, x0 = None, max_it = 50):
    """ Landweber & Cimmino iterative solver
    
    As in Eq. (6.1b) of Discrete Inverse Problems
    """
    if x0 is None:
        x0 = np.zeros(A.shape[1], dtype = complex)
    Ah = np.conj(A.T) # Hermitian of A
    AhA_norm = np.linalg.norm(Ah@A)
    # Check omega
    if omega<=0 or omega>=2/AhA_norm:
        raise ValueError(r"$\omega$ must be a value between 0 and {}".format(2/AhA_norm))
    # matrix D
    D = np.zeros(A.shape[0])
    for row in range(A.shape[0]):
        if not np.any(A[row,:]):
            D[row] = 0
        else:
            row_norm = np.linalg.norm(A[row,:])
            D[row] = 1/(A.shape[0]*row_norm**2)
    # initialize solution vector
    x_sol = np.zeros((A.shape[1], max_it), dtype = complex)
    xk = np.copy(x0)
    # loop
    for k in range(max_it):
        residual = b - A @ xk
        xk += omega * Ah @ np.diag(D) @ residual
        x_sol[:,k] = xk
    return x_sol

def art_solver(A, b, x0 = None, max_it = 50):
    """ Kaczmarz’s method or Algebraic Reconstruction Technique (ART)
    
    As in Sec 6.1.2 of Discrete Inverse Problems
    """
    if x0 is None:
        x0 = np.zeros(A.shape[1], dtype = complex)
    # initialize solution vector
    x_sol = np.zeros((A.shape[1], max_it), dtype = complex)
    xk = np.copy(x0)
    # loop through iterations
    for k in range(max_it):
        # loop through rows of A
        for i in range(A.shape[0]):
            a_i = A[i,:]
            scale = (b[i]-np.conj(a_i) @ xk)/(np.linalg.norm(a_i)**2)
            xk += scale * a_i
        x_sol[:,k] = xk
    return x_sol

def cgls(A, b, x0 = None, max_it = 50):
    """ Conjugate Gradient Least-Squares
    
    As in Sec 6.3.2 of Discrete Inverse Problems
    """
    if x0 is None:
        x0 = np.zeros(A.shape[1], dtype = complex)
    Ah = np.conj(A.T) # Hermitian of A
    # initialize solution vector
    x_sol = np.zeros((A.shape[1], max_it), dtype = complex)
    res_norms = np.zeros(max_it)
    sol_norms = np.zeros(max_it)
    xk = np.copy(x0)
    # loop
    rk = b - A @ xk
    dk = Ah @ rk
    normr2 = np.linalg.norm(dk)**2
    for k in range(max_it):
        # past_residual = residual
        # d_k = Ah @ residual 
        Ad = A @ dk
        alphak = normr2/(np.linalg.norm(Ad)**2)
        xk += alphak * dk
        rk -= alphak * Ad #A @ dk
        sk = Ah @ rk
        normr2_new = np.linalg.norm(sk)**2
        betak = normr2_new/normr2
        normr2 = normr2_new
        dk = sk + betak * dk
        # fill returns
        x_sol[:,k] = xk
        res_norms[k] = np.linalg.norm(rk)
        sol_norms[k] = np.linalg.norm(xk)
    return x_sol, sol_norms, res_norms

def cgls2(A, b, x0 = None, max_it = 50):
    """ Conjugate Gradient Least-Squares
    
    As in Sec 6.3.2 of Discrete Inverse Problems
    """
    if x0 is None:
        x0 = np.zeros(A.shape[1], dtype = complex)
    Ah = np.conj(A.T) # Hermitian of A
    # initialize solution vector
    x_sol = np.zeros((A.shape[1], max_it), dtype = complex)
    res_norms = np.zeros(max_it)
    sol_norms = np.zeros(max_it)
    xk = np.copy(x0)
    # loop
    rk = b - A @ xk
    dk = Ah @ rk
    normr2 = np.linalg.norm(dk)**2;
    for k in range(max_it):
        # Update x and r vectors.
        Ad = A @ dk
        alpha = normr2/(np.conj(Ad.T) @ Ad)
        xk += alpha*dk
        rk -= alpha*Ad
        s = Ah @ rk
        # Update d vector.
        normr2_new = np.linalg.norm(s)**2
        beta = normr2_new/normr2
        normr2 = normr2_new;
        dk = s + beta*dk
        # fill returns
        x_sol[:,k] = xk
        res_norms[k] = np.linalg.norm(rk)
        sol_norms[k] = np.linalg.norm(xk)
    return x_sol, sol_norms, res_norms