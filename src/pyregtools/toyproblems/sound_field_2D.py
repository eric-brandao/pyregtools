"""
This module implements a two-dimensional sound field analysis problem based on
the superposition of propagating plane waves.

The sound field at a given angular frequency, :math:`\\omega` [rad/s], and location
:math:`\\mathbf{r}_{\\text{m}} = (x_{\\text{m}}, z_{\\text{m}})` [m] is 
described by

.. math::

    \\tilde{p}(\\mathbf{r}_{\\text{m}}, \\omega) 
    = \\sum\\limits_{i=1}^{L} \\tilde{x}_i \\exp(\\text{-j} \\mathbf{k}_i \\cdot \\mathbf{r}_{\\text{m}}),
    

where :math:`\\mathbf{k}_i = (k_{x_i}, k_{z_i})` is the i-th plane-wave 
wave-number vector, with norm :math:`\\|\\mathbf{k}_i\\|_2 = \\frac{\\omega}{c_0}=k_0`.

The problem is restricted to propagating plane-waves, which satisfy the condition

.. math::

    \\|\\mathbf{k}_i\\|_2^2 = k_{x}^2 + k_{z}^2 = k_0^2.

Thus, the plane-wave kernel is given by

.. math::

    K(\\mathbf{k}_i\\, \\mathbf{r}_{\\text{m}})
    = \\text{e}^{\\text{-j} \\mathbf{k}_i \\cdot \\mathbf{r}_{\\text{m}}}.

Thus, by measuring at :math:`M = 1, 2, \\ldots, M` locations, the measurement 
vector, at angular frequency :math:`\\omega` becomes

.. math::

    \\mathbf{b} 
    = 
    [\\tilde{p}(\\mathbf{r}_{1}), \\tilde{p}(\\mathbf{r}_{2}), \\ldots, \\tilde{p}(\\mathbf{r}_{M})] \\in \\mathbb{C}^{M \\times 1}.

Then, one can form a set of :math:`L` candidate propagating wave-number vectors, 
distributed along the circunference of radius :math:`k_0`. Thus, the plane-wave kernel 
is given by

.. math::

    K(\\mathbf{k}_i\\, \\mathbf{r}_{\\text{m}})
    = \\exp(\\text{-j} \\mathbf{k}_i \\cdot \\mathbf{r}_{\\text{m}}),

with :math:`i = 1, 2, \\ldots, L`. 

Thus, the problem is discretized, resulting in the linear system

.. math::

    \\mathbf{A}\\mathbf{x} = \\mathbf{b},

where :math:`\\mathbf{x} \\in \\mathbb{C}^{L \\times 1}` represents the complex amplitude of each
plane-wave, and :math:`\\mathbf{A} \\in \\mathbb{C}^{M \\times L}` is the problem sensing matrix.

The formulation of such a problem in 3D can be found in Refs. [1]_ and [2]_. For versions of the
problem including evanescent waves, see [3]_ and [4]_.

References
----------
.. [1] Nolan, Fernandez-Grande, E., Brunskog, J. and Jeong, C-H., *A wavenumber 
    approach to quantifyingthe isotropy of the sound field in reverberant spaces*, 
    The Journal of the Acoustical Society of America, 143(4), 2018, p.2514--2526.

.. [2] Nolan, M. Verburg, S. A., Brunskog, J. and Fernandez-Grande, E., *Experimental 
    characterization of the sound field in a reverberation room*, 
    The Journal of the Acoustical Society of America, 145(4), 2019, p.2237--2246.

.. [3] Brandão, E. and Fernandez-Grande, E., *Analysis of the sound field above finite 
    absorbers in the wave-number domain*, 
    The Journal of the Acoustical Society of America, 151(5), 2022, p.3019--3030.

.. [4] Hald, J., *Basic theory and properties of statistically optimized near-field 
    acoustical holography*, 
    The Journal of the Acoustical Society of America, 125(4), 2009, p.2105--2120.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import warnings
from pyregtools.utils import add_gaussian_noise

class SoundField2D:
    """
    Two-dimensional sound field simulation and plane-wave estimation.

    Represents a two-dimensional sound field as a superposition of
    propagating plane waves at a single frequency.

    The class supports two complementary formulations:

    - Forward simulation, in which the propagation directions and
      complex amplitudes of the plane waves are prescribed.

    - Inverse estimation, in which a dictionary of candidate plane-wave
      directions is used to construct a sensing matrix for estimating
      the unknown complex amplitudes.

    The sound pressure is represented by

    .. math::

        \\tilde{p}(\\mathbf{r})
        =
        \\sum_{i=1}^{N}
        \\tilde{a}_i
        \\exp(-\\mathrm{j}\\mathbf{k}_i\\cdot\\mathbf{r}),

    where :math:`\\tilde{a}_i` is the complex amplitude of the
    :math:`i`-th plane wave, :math:`\\mathbf{k}_i` is its wave-number
    vector, and :math:`\\mathbf{r}=(x,z)` is the observation position.
    
    .. list-table::
       :widths: 20 20 60
       :header-rows: 1

       * - Attribute
         - Shape
         - Description
       * - ``c0``
         - scalar
         - Speed of sound in the medium, in m/s.
       * - ``freq``
         - scalar
         - Frequency of analysis, in Hz.
       * - ``k0``
         - scalar
         - Acoustic wavenumber, in rad/m.
       * - ``coord``
         - ``(M, 2)``
         - Microphone coordinates, in meters.
       * - ``b_true``
         - ``(M,)``
         - Simulated complex sound-pressure measurements, in Pa.
       * - ``b_noisy``
         - ``(M,)``
         - Simulated complex sound-pressure measurements with added noise, in Pa.
       * - ``nwaves``
         - scalar
         - Number of candidate plane-wave directions, :math:`L`.
       * - ``A``
         - ``(M, L)``
         - Complex sensing matrix for the inverse problem.

    The attributes ``c0``, ``freq``, and ``k0`` are initialized
    when the class is instantiated. The remaining attributes are
    created by :meth:`set_mic_array`, :meth:`compute_pres`,
    :meth:`add_noise`, or :meth:`get_sens_mtx`, as appropriate.     
         
    Parameters
    ----------
    c0 : float, default=340
        Speed of sound in the medium, in m/s.
    freq : float, default=1000
        Frequency of analysis, in Hz.
    """
    def __init__(self, c0 = 340, freq = 1000):
        self.c0 = c0
        self.freq = freq
        self.k0 = 2*np.pi*self.freq /self.c0

    def pt_array(self, x_len = 0.5, z_len = 0.5, n_x = 5, n_z =5):
        """
        Generate a rectangular grid of spatial coordinates.

        Creates a uniformly spaced grid in the :math:`xz` plane, centered
        at the origin. The grid extends over

        .. math::

            -\\frac{l_x}{2} \\leq x \\leq \\frac{l_x}{2},
            \\qquad
            -\\frac{l_z}{2} \\leq z \\leq \\frac{l_z}{2}.

        The grid can be used to define microphone positions or spatial
        coordinates for sound-field reconstruction.

        Parameters
        ----------
        x_len : float, default=0.5
            Length of the grid along the :math:`x` direction, in meters.
        z_len : float, default=0.5
            Length of the grid along the :math:`z` direction, in meters.
        n_x : int, default=5
            Number of grid points along the :math:`x` direction.
        n_z : int, default=5
            Number of grid points along the :math:`z` direction.

        Returns
        -------
        coord : ndarray, shape (M, 2)
            Spatial coordinates of the grid points, where
            :math:`M = n_x n_z`. Each row contains the coordinates
            :math:`(x_m, z_m)` of one point.
        """
        xc = np.linspace(-x_len/2, x_len/2, n_x)
        zc = np.linspace(-z_len/2, z_len/2, n_z)
        # meshgrid
        x_grid, z_grid = np.meshgrid(xc, zc)
        # initialize receiver list in memory
        coord = np.zeros((n_x * n_z, 2))
        coord[:, 0] = x_grid.flatten()
        coord[:, 1] = z_grid.flatten()
        return coord
    
    def set_mic_array(self, x_len = 0.5, z_len = 0.5, n_x = 5, n_z =5):
        """
        Define a rectangular microphone array.

        Generates a uniformly spaced rectangular microphone array in the
        :math:`xz` plane, centered at the origin.

        Parameters
        ----------
        x_len : float, default=0.5
            Array length along the :math:`x` direction, in meters.
        z_len : float, default=0.5
            Array length along the :math:`z` direction, in meters.
        n_x : int, default=5
            Number of microphones along the :math:`x` direction.
        n_z : int, default=5
            Number of microphones along the :math:`z` direction.

        Notes
        -----
        The microphone coordinates are generated using :meth:`pt_array`
        and stored in ``coord``, an array of shape ``(M, 2)``, where
        :math:`M = n_x n_z`.
        """
        self.coord = self.pt_array(x_len = x_len, z_len = z_len, 
                                   n_x = n_x, n_z = n_z)
    
    def get_k(self, theta_deg = 90):
        """
        Compute the wavenumber vector of a propagating plane wave.

        The wavenumber vector is defined as

        .. math::

            \\mathbf{k} =
            k_0
            \\begin{bmatrix}
                \\sin(\\theta) \\\\
                \\cos(\\theta)
            \\end{bmatrix},

        where :math:`k_0 = 2\\pi f/c_0` is the acoustic wavenumber.
        The angle :math:`\\theta` is measured from the positive
        :math:`z`-axis toward the positive :math:`x`-axis.

        Parameters
        ----------
        theta_deg : float, default=90
            Plane-wave angle in degrees.

        Returns
        -------
        k_vec : ndarray, shape (2,)
            Wavenumber vector ``[kx, kz]``, in rad/m.
        """
        kx = self.k0*np.sin(np.deg2rad(theta_deg))
        kz = self.k0*np.cos(np.deg2rad(theta_deg))
        return np.array([kx,kz])
    
    def compute_pres(self, theta_deg = (45, -45), amps = (1, 0.7)):
        """
        Simulate the complex sound pressure at the microphone positions.

        Computes the superposition of :math:`N` propagating plane waves
        with prescribed propagation angles and complex amplitudes:

        .. math::

            \\tilde{p}(\\mathbf{r}_m)
            =
            \\sum_{i=1}^{N}
            \\tilde{a}_i
            \\exp(-\\mathrm{j}\\mathbf{k}_i\\cdot\\mathbf{r}_m),

        where :math:`\\mathbf{r}_m` is the position of microphone
        :math:`m`, :math:`\\tilde{a}_i` is the complex amplitude of
        plane wave :math:`i`, and :math:`\\mathbf{k}_i` is its
        wavenumber vector.

        Parameters
        ----------
        theta_deg : array_like, default=(45, -45)
            Propagation angles of the plane waves, in degrees.
            The number of angles determines :math:`N`.
        amps : array_like, default=(1, 0.7)
            Complex amplitudes of the plane waves, in Pa.
            Must have the same number of elements as ``theta_deg``.

        Notes
        -----
        The microphone positions must first be defined using
        :meth:`set_mic_array`.

        The simulated pressure is stored in ``b_true``, a complex
        array of shape ``(M,)``, where :math:`M` is the number of
        microphones. No noise is added by this method.
        """
        if len(amps) != len(theta_deg):
            raise ValueError("amplitude and theta_deg must have the same number of elements.")
        self.b_true = np.zeros(self.coord.shape[0], dtype = complex)
        for jt, theta in enumerate(theta_deg):
            k_vec = self.get_k(theta_deg = theta)
            self.b_true += amps[jt]*np.exp(-1j*self.coord @ k_vec)

    def smooth_sf(self, factor = 1):
        """
        Simulate a sound field with a smooth angular amplitude distribution.

        Generates :math:`N=360` uniformly spaced plane-wave directions
        over the interval :math:`[0, 2\\pi)` and assigns their amplitudes
        according to

        .. math::

            \\tilde{a}_i = \\sin^2(q\\theta_i),

        where :math:`q` is the amplitude modulation factor and
        :math:`\\theta_i = 2\\pi i/N`, with :math:`i=0,\\ldots,N-1`.

        The resulting sound pressure is computed by superposing the
        plane waves using :meth:`compute_pres`.

        Parameters
        ----------
        factor : float, default=1
            Angular modulation factor, :math:`q`, controlling the
            number and angular spacing of the amplitude lobes.

        Notes
        -----
        The simulated complex pressure is stored in ``b_true``.

        The amplitudes are real and nonnegative, with a maximum
        value of one. For positive integer values of ``factor``,
        the distribution contains ``2 * factor`` lobes over
        a full revolution.
        """
        theta = 2 * np.pi * np.arange(360) / 360
        amps = np.sin(factor * theta)**2
        self.compute_pres(theta_deg = np.rad2deg(theta), amps = amps)
            
    def add_noise(self, snr=30, seed=0):
        """
        Add complex Gaussian noise to the simulated sound pressure.

        Uses :func:`~pyregtools.utils.reg_utils.add_gaussian_noise`
        to generate noisy measurements at the prescribed SNR.

        Parameters
        ----------
        snr : float, default=30
            Target signal-to-noise ratio in decibels.
        seed : int, default=0
            Random seed.

        Notes
        -----
        The noisy measurement vector is stored in ``b_noisy``.
        """
        self.b_noisy = add_gaussian_noise(
            self.b_true, snr=snr, seed=seed)       
    
    def get_sens_mtx(self, nwaves = 180):
        """
        Construct the plane-wave sensing matrix.

        Generates a dictionary of :math:`L` uniformly distributed
        propagating plane-wave directions and evaluates the plane-wave
        kernel at the :math:`M` microphone positions.

        The sensing matrix is defined by

        .. math::

            A_{mi}
            =
            \\exp(-\\mathrm{j}\\mathbf{k}_i\\cdot\\mathbf{r}_m),

        where :math:`\\mathbf{r}_m` is the position of microphone
        :math:`m` and :math:`\\mathbf{k}_i` is the wavenumber vector
        associated with candidate direction :math:`i`.

        The resulting inverse problem is

        .. math::

            \\mathbf{A}\\mathbf{x} = \\mathbf{b},

        where :math:`\\mathbf{x}\\in\\mathbb{C}^{L}` contains the
        unknown complex plane-wave amplitudes.

        Parameters
        ----------
        nwaves : int, default=180
            Number of candidate plane-wave directions, :math:`L`. Must be :math:`>1`

        Notes
        -----
        The microphone positions must first be defined using
        :meth:`set_mic_array`.

        The candidate directions are generated using
        :meth:`wave_directions`. The sensing matrix is stored in
        ``A``, a complex array of shape ``(M, L)``. The number of
        candidate directions is stored in ``nwaves``.

        This method constructs the forward operator for the inverse
        problem; it does not estimate the unknown amplitudes.
        """
        if not isinstance(nwaves, (int, np.integer)) or nwaves <= 1:
            raise ValueError("nwaves must be a positive integer.")
        self.nwaves = nwaves
        theta_dir = self.wave_directions()
        k_sens_vec = self.k0 * np.array([np.sin(theta_dir), np.cos(theta_dir)])
        self.A = np.exp(-1j * self.coord @ k_sens_vec)

    def wave_directions(self,):
        """
        Generate uniformly distributed candidate plane-wave directions.

        The angular dictionary contains :math:`L` directions given by

        .. math::

            \\theta_i = \\frac{2\\pi i}{L},
            \\qquad i = 0, 1, \\ldots, L-1,

        where :math:`L` is the number of candidate plane waves.
        The directions cover the interval :math:`[0, 2\\pi)` without
        repeating the endpoint.

        Returns
        -------
        theta_dir : ndarray, shape (L,)
            Candidate plane-wave angles, in radians.

        Notes
        -----
        The number of candidate directions is stored in ``nwaves``,
        defined when :meth:`get_sens_mtx` is called.
        """
        theta_dir = 2 * np.pi * np.arange(self.nwaves) / self.nwaves
        return theta_dir

    def reconstruct_pres(self, x_sol, x_len = 0.5, z_len = 0.5, n_x = 20, n_z = 20):
        """
        Reconstruct the sound pressure from estimated plane-wave amplitudes.

        Evaluates the plane-wave expansion at a rectangular grid of
        reconstruction points:

        .. math::

            \\tilde{p}_{\\mathrm{rec}}(\\mathbf{r}_m)
            =
            \\sum_{i=1}^{L}
            \\tilde{x}_{\\mathrm{sol},i}
            \\exp(-\\mathrm{j}\\mathbf{k}_i\\cdot\\mathbf{r}_m).

        The reconstruction is computed as

        .. math::

            \\mathbf{p}_{\\mathrm{rec}}
            =
            \\mathbf{A}_{\\mathrm{rec}}\\mathbf{x}_{\\mathrm{sol}},

        where :math:`\\mathbf{A}_{\\mathrm{rec}}` is the sensing matrix
        evaluated at the reconstruction coordinates.

        Parameters
        ----------
        x_sol : array_like, shape (L,)
            Estimated complex amplitudes of the candidate plane waves.
        x_len : float, default=0.5
            Length of the reconstruction grid along the :math:`x`
            direction, in meters.
        z_len : float, default=0.5
            Length of the reconstruction grid along the :math:`z`
            direction, in meters.
        n_x : int, default=20
            Number of reconstruction points along the :math:`x` direction.
        n_z : int, default=20
            Number of reconstruction points along the :math:`z` direction.

        Returns
        -------
        coord_recon : ndarray, shape (P, 2)
            Coordinates of the reconstruction points, where
            :math:`P = n_x n_z`.
        pres_recon : ndarray, shape (P,)
            Reconstructed complex sound pressure, in Pa.

        Notes
        -----
        The plane-wave dictionary must first be defined using
        :meth:`get_sens_mtx`.

        The reconstruction uses the same candidate directions and
        frequency as the sensing matrix, but evaluates the plane-wave
        kernel at the new coordinates generated by :meth:`pt_array`.
        """
        # reconstruction coords
        coord_recon = self.pt_array(x_len = x_len, z_len = z_len, 
                                  n_x = n_x, n_z = n_z)
        # reconstruction matrix
        theta_dir = self.wave_directions()
        k_sens_vec = self.k0 * np.array([np.sin(theta_dir), np.cos(theta_dir)])
        Ar = np.exp(-1j * coord_recon @ k_sens_vec)
        pres_recon = Ar @ x_sol
        return coord_recon, pres_recon

    def plot_measured_field(self, coord, pres):
        """
        Plot the real part of the sound pressure at discrete positions.

        Creates a scatter plot of the real part of the complex sound
        pressure in the :math:`xz` plane. The color scale is symmetric
        about zero, with limits determined by the maximum absolute
        value of the real pressure.

        Parameters
        ----------
        coord : array_like, shape (M, 2)
            Spatial coordinates of the measurement points, in meters.
            Each row contains the coordinates :math:`(x_m, z_m)`.
        pres : array_like, shape (M,)
            Complex sound pressure at the measurement points, in Pa.

        Notes
        -----
        Only the real part of ``pres`` is displayed. The color scale
        extends from :math:`-p_{\\max}` to :math:`p_{\\max}`, where

        .. math::

            p_{\\max}
            =
            \\max_m |\\operatorname{Re}\\{p_m\\}|.

        The number of points, :math:`M`, is displayed in the plot title.
        """
        min_val, max_val = np.real(pres).min(), np.real(pres).max()
        range_c = np.amax(np.abs([min_val, max_val]))
        plt.figure(figsize=(5, 4))
        scatter = plt.scatter(coord[:,0], coord[:,1], c=np.real(pres), 
                                cmap='bwr', s=30, edgecolor='none',
                                vmin = -range_c, vmax = range_c)
        cbar = plt.colorbar(scatter)
        cbar.set_label(r'$\mathcal{R}\left\{p\right\}$', fontsize=10)
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$z$ [m]")
        plt.title(r'$M = {}$.'.format(coord.shape[0]))
        plt.tight_layout()
            
    def plot_solution_field(self, x_sol, normalize = True):
        """
        Plot the estimated plane-wave amplitudes on the unit circle.

        Displays the magnitude of the estimated complex amplitudes at
        their corresponding candidate propagation directions. Each
        direction is represented by a point on the unit circle:

        .. math::

            \\mathbf{u}_i =
            \\begin{bmatrix}
                \\sin(\\theta_i) \\\\
                \\cos(\\theta_i)
            \\end{bmatrix}.

        The color of each point represents the magnitude of the
        corresponding plane-wave amplitude.

        Parameters
        ----------
        x_sol : array_like, shape (L,)
            Estimated complex plane-wave amplitudes, in Pa.
        normalize : bool, default=True
            If True, normalize the amplitude magnitudes by their maximum
            value before plotting. Otherwise, display the magnitudes in Pa.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the plane-wave amplitude plot.
        ax : matplotlib.axes.Axes
            Axes containing the scatter plot.

        Notes
        -----
        The candidate directions must first be defined using
        :meth:`get_sens_mtx`.

        Normalization is applied only to the plotted values and does
        not modify ``x_sol``.
        """
        if normalize:
            x_sol2plot = x_sol/np.amax(np.abs(x_sol))
        else:
            x_sol2plot = x_sol
        # scatter - coord
        theta_dir = self.wave_directions()
        coords_sol = np.array([np.sin(theta_dir), np.cos(theta_dir)]).T
        
        plt.figure(figsize=(5, 4))
        scatter = plt.scatter(coords_sol[:,0], coords_sol[:,1], 
                              c=np.abs(x_sol2plot), cmap='inferno', s=30, edgecolor='none',
                              vmin = 0, vmax = np.abs(x_sol2plot).max())
        cbar = plt.colorbar(scatter)
        cbar.set_label(r'$|x|$', fontsize=10)
        plt.xlabel(r"$k_x/k_0$ [-]")
        plt.ylabel(r"$k_z/k_0$ [-]")
        plt.tight_layout()