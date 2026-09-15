""" One dimensional gravity survey problem

The idea here is to measure the acceleration of gravity, :math:`g(x^{\\prime})`, 
at a distance :math:`d`. A burried source density field :math:`\\rho(x)` originates
the gravity acceleration that is measured. The task is to recover the density field.
"""
import numpy as np
import scipy
import warnings
from pyregtools.utils import add_gaussian_noise


class Gravity1D():
    """ One dimensional gravity survey problem
    """

    def __init__(self, lb = 0, ub = 1, d = 0.5):
        """ class init
        
        Parameters:
        ------------
        lb : float
            lower limit of integration.
        ub : float
            upper limit of integration.
        d : float
            vertical distance from measurement to source term.
    """
        self.lb, self.ub = lb, ub
        self.d = d

    def sample_rho_uniform(self, L = 20):
        """ Sample the density source term (uniformly)

        Parameters:
        ------------
        L : int 
            The number of source points in space
        """
        self.x = np.linspace(self.lb, self.ub, num = L)
        self.weights = ((self.ub-self.lb)/L) * np.ones(L)
        #self.x_true = np.zeros(L)

    def sample_rho_gauss_legendre(self, L = 20):
            """ Sample the density source term using Gauss-Legendre quadrature
    
            Parameters:
            ------------
            L : int 
                The number of source points in space
            """
            roots, self.weights = scipy.special.roots_legendre(L) # roots and weights of G-L polynomials
            self.x = ((self.ub - self.lb) / 2) * roots + ((self.lb + self.ub) / 2) 

    def sample_g(self, M = 20):
        """ Sample the measurement of the acceleration

        Parameters:
        ------------
        M : int 
            The number of measurement points
        """
        self.coord = np.linspace(self.lb, self.ub, num = M)

    def kernel(self, x, xl):
        """ kernel (integrand value)

        Parameters:
        ------------
        x : float or ndarray 
            coordinates along the source term
        xl : float or ndarray 
            coordinates along the measurement
        """
        k = (self.d/(self.d**2 + (x-xl)**2)**(3/2))
        return k

    def sens_mtx(self, ):
        """ Build sensing matrix
        """
        M, L = self.coord.shape[0], self.x.shape[0]
        self.A = np.zeros((M,L))
        for jc, coord in enumerate(self.coord):
            self.A[jc, :] = self.weights*self.kernel(self.x, coord)

    def density_piecewise(self, ):
        """ Piecewise source density field
        
        Defines a function of order L for which the first 1/3 of points are 2.0, 
        and the rest are 1.0.
        
        """
        L = self.x.shape[0]
        self.x_true = np.ones(L) # remaining part = 1.0
        self.x_true[:int(L/3)] = 2 # first 1/3 = 2.0

    def density_sin(self, amplitude = 1.0, freq = 1.0):
        """ Sinusoidal source density field
        
        Defines a function or order len(x) for a sinusoidal source function.
        
        Parameters:
        -----------
        amplitude : float
            amplitude of the phenomenon
        freq : float
            spatial frequency of the phenomenon
        
        """
        self.x_true = amplitude*np.sin(2*np.pi*freq*self.x)

    def density_rechann(self, x_start = 0.5):
        """ Rectangular + hanning density function
        
        Parameters:
        ------------
        x_start : float
            where window starts
        """
        L = len(self.x)
        nh = len(self.x[self.x > x_start])
        hw = scipy.signal.windows.hann(2*nh)
        self.x_true = 2*np.ones(L)
        self.x_true[L-nh:] = hw[nh:]+1

    def multi_density_sin(self, amplitude = (1.0, 0.5), freq = (0.5, 1.0)):
        """ Sinusoidal source density field
        
        Defines a function or order len(x) for a sinusoidal source function.
        
        Parameters:
        -----------
        amplitude : ndarray
            amplitudes of the phenomenon
        freq : ndarray
            spatial frequencies of the phenomenon
        
        """
        self.x_true = np.zeros(len(self.x))
        for jf, f in enumerate(freq):
            self.x_true += amplitude[jf]*np.sin(2*np.pi*f*self.x)

    def noiseless_meas(self, ):
        """ Compute noiseless measurement
        """
        self.b_true = self.A @ self.x_true

    def add_noise(self, snr = 25, seed = 0):
        """ Add Gaussian noise to a noiseless measurement
        """
        self.b_noisy = add_gaussian_noise(self.b_true, snr = snr, seed = seed)