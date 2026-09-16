"""
This module implements a one-dimensional gravity survey problem based on
the integral equation

.. math::

    g(x^{\\prime})
    =
    \\int_a^b
    K(x, x^{\\prime})\\rho(x)\\,dx,

where :math:`\\rho(x)` is the unknown source density distribution and
:math:`g(x^{\\prime})` is the gravitational acceleration measured at
coordinate :math:`x^{\\prime}` and vertical distance :math:`d` from the
source.

The gravity kernel is given by

.. math::

    K(x, x^{\\prime})
    =
    \\frac{d}
    {\\left[d^2 + (x-x^{\\prime})^2\\right]^{3/2}}.

The integral equation is discretized using either the midpoint rule or
Gauss--Legendre quadrature, resulting in the linear system

.. math::

    \\mathbf{A}\\mathbf{x} = \\mathbf{b},

where :math:`\\mathbf{x}` represents the discretized source density and
:math:`\\mathbf{b}` contains the gravity measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import warnings
from pyregtools.utils import add_gaussian_noise


class Gravity1D:
    """
    One-dimensional gravity survey inverse problem.

    Represents a gravity survey in which an unknown density distribution
    is defined over the interval :math:`[a,b]` and the resulting
    gravitational acceleration is measured at a vertical distance
    :math:`d`.

    The source domain can be discretized using the midpoint rule or
    Gauss--Legendre quadrature. The class provides methods for constructing
    the sensing matrix, defining several true source distributions, and
    generating noiseless and noisy measurements.

    .. list-table::
       :header-rows: 1
       :widths: 20 25 55

       * - Attribute
         - Shape
         - Description
       * - ``lb``
         - ``float``
         - Lower bound of integration.
       * - ``ub``
         - ``float``
         - Upper bound of integration.
       * - ``d``
         - ``float``
         - Vertical measurement distance.
       * - ``x``
         - ``(L,)``
         - Source coordinates.
       * - ``weights``
         - ``(L,)``
         - Quadrature weights.
       * - ``coord``
         - ``(M,)``
         - Measurement coordinates.
       * - ``A``
         - ``(M, L)``
         - Sensing matrix.
       * - ``x_true``
         - ``(L,)``
         - True source density.
       * - ``b_true``
         - ``(M,)``
         - Noiseless measurements.
       * - ``b_noisy``
         - ``(M,)``
         - Noisy measurements.

    Parameters
    ----------
    lb : float, default=0
        Lower bound of the source interval.
    ub : float, default=1
        Upper bound of the source interval.
    d : float, default=0.5
        Vertical distance between the measurement line and the source
        distribution.
    
    The attributes associated with discretization, source density, and measurements 
    are created when their corresponding methods are called.
    """

    def __init__(self, lb = 0, ub = 1, d = 0.5):
        self.lb, self.ub = lb, ub
        self.d = d

    def sample_rho_mp(self, L = 32):
        """
        Discretize the source density using the midpoint rule.

        Divides the source interval :math:`[a,b]` into :math:`L` equally
        spaced cells and places one source point at the center of each cell.
        The corresponding quadrature weights are

        .. math::

            w_l = \\Delta x,
            \\qquad
            \\Delta x = \\frac{b-a}{L}.

        Parameters
        ----------
        L : int, default=32
            Number of source points.

        Notes
        -----
        This method defines the source coordinates ``x`` and quadrature
        weights ``weights`` used to discretize the gravity integral.
        """
        dx = (self.ub - self.lb) / L
        self.x = self.lb + (np.arange(L) + 0.5) * dx
        self.weights = dx * np.ones(L)

    def sample_rho_gauss_legendre(self, L = 32):
        """
        Discretize the source density using Gauss--Legendre quadrature.

        Computes :math:`L` Gauss--Legendre nodes and weights on
        :math:`[-1,1]` and maps them to the source interval :math:`[a,b]`.
        The transformed source coordinates are

        .. math::

            x_l =
            \\frac{b-a}{2}\\xi_l + \\frac{a+b}{2},

        where :math:`\\xi_l` are the Gauss--Legendre nodes. The corresponding
        quadrature weights are

        .. math::

            w_l = \\frac{b-a}{2}\\hat{w}_l,

        where :math:`\\hat{w}_l` are the Gauss--Legendre weights on
        :math:`[-1,1]`.

        Parameters
        ----------
        L : int, default=32
            Number of source points and quadrature nodes.

        Notes
        -----
        This method defines the source coordinates ``x`` and quadrature
        weights ``weights`` used to discretize the gravity integral.
        """
        roots, weights = scipy.special.roots_legendre(L) # roots and weights of G-L polynomials
        self.weights = 0.5*(self.ub-self.lb)*weights
        self.x = ((self.ub - self.lb) / 2) * roots + ((self.lb + self.ub) / 2) 

    def sample_g(self, M = 32):
        """
        Define the gravity measurement coordinates.

        Places :math:`M` equally spaced measurement points over the interval
        :math:`[a,b]`, including both endpoints.

        Parameters
        ----------
        M : int, default=32
            Number of gravity measurement points.

        Notes
        -----
        The measurement coordinates are stored in ``coord`` and are used
        when constructing the sensing matrix.
        """
        self.coord = np.linspace(self.lb, self.ub, num = M)

    def kernel(self, x, xl, d):
        """
        Evaluate the gravity kernel.

        Computes the kernel of the one-dimensional gravity integral equation,

        .. math::

            K(x, x^{\\prime})
            =
            \\frac{d}
            {\\left[d^2 + (x-x^{\\prime})^2\\right]^{3/2}},

        where :math:`d` is the vertical distance between the source and
        measurement lines.

        Parameters
        ----------
        x : float or ndarray
            Source coordinate(s).
        xl : float or ndarray
            Measurement or reconstruction coordinate(s).
        d : float, 
            Vertical distance between the measurement line and the source
            distribution.

        Returns
        -------
        k : float or ndarray
            Gravity kernel evaluated at the specified coordinates.
            The output shape follows NumPy broadcasting rules.
        """
        k = (self.d/(self.d**2 + (x-xl)**2)**(3/2))
        return k

    def sens_mtx(self, ):
        """
        Construct the gravity sensing matrix.

        Discretizes the gravity integral equation using the source
        coordinates and quadrature weights previously defined by one of
        the source-sampling methods.

        The matrix entries are given by

        .. math::

            A_{ml}
            =
            w_l K(x_l, x_m^{\\prime}),

        where :math:`w_l` denotes the quadrature weight associated with
        source coordinate :math:`x_l`, and :math:`x_m^{\\prime}` is the
        measurement coordinate.

        The resulting sensing matrix satisfies

        .. math::

            \\mathbf{b}_{\\mathrm{true}}
            =
            \\mathbf{A}\\mathbf{x}_{\\mathrm{true}},

        where :math:`\\mathbf{A} \\in \\mathbb{R}^{M \\times L}`,
        :math:`\\mathbf{x}_{\\mathrm{true}} \\in \\mathbb{R}^{L}`,
        and :math:`\\mathbf{b}_{\\mathrm{true}} \\in \\mathbb{R}^{M}`.

        Notes
        -----
        The sensing matrix is stored in the ``A`` attribute.

        The source coordinates and quadrature weights must be initialized
        using ``sample_rho_mp`` or ``sample_rho_gauss_legendre``. The
        measurement coordinates must be initialized using ``sample_g``.
        """
        M, L = self.coord.shape[0], self.x.shape[0]
        self.A = np.zeros((M,L))
        for jc, coord in enumerate(self.coord):
            self.A[jc, :] = self.weights*self.kernel(self.x, coord, self.d)

    def density_piecewise(self, ):
        """
        Define a piecewise-constant source density.

        Constructs a discrete source density with value 2.0 at the first
        :math:`\\lfloor L/3 \\rfloor` source points and value 1.0 at the
        remaining points, where :math:`L` is the number of source points.

        The resulting density vector is stored in ``x_true``.
        """
        L = self.x.shape[0]
        self.x_true = np.ones(L) # remaining part = 1.0
        self.x_true[:int(L/3)] = 2 # first 1/3 = 2.0

    def density_sin(self, amplitude = 1.0, kx = 6.28):
        """
        Define a sinusoidal source density.

        Evaluates the source density at the previously defined source
        coordinates according to

        .. math::

            \\rho(x) = A\\sin(k_x x),

        where :math:`A` is the amplitude and :math:`k_x` is the spatial
        frequency in [rad/m].

        Parameters
        ----------
        amplitude : float, default=1.0
            Amplitude of the sinusoidal density.
        kx : float, default= :math:`2\\pi`
            Spatial frequency of the sinusoidal density, in [rad/m].

        Notes
        -----
        The resulting density vector is stored in ``x_true``.
        """
        self.x_true = amplitude*np.sin(kx*self.x)

    def multi_density_sin(self, amplitude = (1.0, 0.5), kx = (3.14, 6.28)):
        """
        Define a source density as a sum of sinusoidal components.

        Evaluates the source density according to

        .. math::

            \\rho(x)
            =
            \\sum_{j=1}^{J}
            A_j\\sin(k_{xj} x),

        where :math:`A_j` and :math:`k_{xj}` are the amplitude and spatial
        frequency ([rad/m]) of the :math:`j`-th component, respectively.

        Parameters
        ----------
        amplitude : array_like, default=(1.0, 0.5)
            Amplitudes of the sinusoidal components.
        kx : array_like, default=(:math:`\\pi`, :math:`2\\pi`)
            Spatial frequencies of the sinusoidal components, in [rad/m]. 
            Must have the same length as ``amplitude``.

        Notes
        -----
        The resulting density vector is stored in ``x_true``.
        """
        if len(amplitude) != len(kx):
            raise ValueError("amplitude and kx must have the same number of elements.")
        self.x_true = np.zeros(len(self.x))
        for jk, k in enumerate(kx):
            self.x_true += amplitude[jk]*np.sin(k*self.x)

    def density_hann_transition(self, x_start = 0.5):
        """
        Define a piecewise density with a Hann-window transition.

        Initializes the source density to 2.0 and replaces its final
        samples with the descending half of a Hann window shifted
        upward by 1.0.

        The number of samples in the transition is determined by the
        number of source coordinates greater than ``x_start``.

        Parameters
        ----------
        x_start : float, default=0.5
            Coordinate used to determine the beginning of the
            Hann-window transition. If ``x_start`` is larger than
            the maximum value of ``x``, then ``x_start`` is set to half of
            the maximum value of ``x``.

        Notes
        -----
        The resulting density vector is stored in ``x_true``.
        """
        if x_start >= self.x.max():
            x_start = 0.5*self.x.max()
        L = len(self.x)
        nh = len(self.x[self.x > x_start])
        hw = scipy.signal.windows.hann(2*nh)
        self.x_true = 2*np.ones(L)
        self.x_true[L-nh:] = hw[nh:]+1

    def noiseless_meas(self, ):
        """
        Compute the noiseless gravity measurements.

        Evaluates the discrete forward model

        .. math::

            \\mathbf{b}_{\\mathrm{true}}
            =
            \\mathbf{A}\\mathbf{x}_{\\mathrm{true}},

        where :math:`\\mathbf{A}` is the sensing matrix and
        :math:`\\mathbf{x}_{\\mathrm{true}}` is the prescribed source
        density vector.

        Notes
        -----
        The resulting measurement vector, of shape ``(M,)``, is stored
        in ``b_true``.
        """
        self.sens_mtx()
        self.b_true = self.A @ self.x_true

    def add_noise(self, snr = 25, seed = 0):
        """
        Generate noisy gravity measurements.

        Adds Gaussian noise to the noiseless measurement vector according
        to the observation model

        .. math::

            \\mathbf{b}
            =
            \\mathbf{b}_{\\mathrm{true}} + \\mathbf{n},

        where :math:`\\mathbf{n}` is a Gaussian noise vector scaled to
        achieve the prescribed signal-to-noise ratio (SNR).

        Parameters
        ----------
        snr : float, default=25
            Target signal-to-noise ratio in decibels (dB).
        seed : int, default=0
            Seed used for reproducible noise generation.

        Notes
        -----
        The noiseless measurement vector must be computed beforehand
        using ``noiseless_meas``.

        Noise generation is performed by :func:`~pyregtools.utils.reg_utils.add_gaussian_noise`.

        The resulting noisy measurement vector, of shape ``(M,)``, is
        stored in ``b_noisy``.
        """
        self.b_noisy = add_gaussian_noise(self.b_true, snr = snr, seed = seed)

    def plot_problem(self, plot_in_color = True):
        """
        Plot a schematic of the gravity survey geometry.

        Displays the source and measurement lines, their discretization
        points, and the vertical separation between them.

        The source coordinates, measurement coordinates, and separation
        must be defined beforehand.

        Parameters
        ----------
        plot_in_color : bool, default=True
            If ``True``, use colored points in the plot. If ``False``, use a
            grayscale representation.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Matplotlib figure.
        ax : matplotlib.axes.Axes
            Matplotlib axes.
        """
        if plot_in_color:
            colors = ['royalblue', 'r']
        else:
            colors = ['k', 'grey']

        fig, ax = plt.subplots(figsize=(10, 4))

        # Horizontal extent of the schematic
        span = self.ub - self.lb
        margin = 0.12 * span

        # Source and measurement lines
        ax.plot([self.lb, self.ub], [0, 0], 'k-', lw=1)
        ax.plot([self.lb, self.ub], [self.d, self.d], 'k-', lw=1)

        # Discretization points
        ax.plot(self.x, np.zeros_like(self.x), 's', color = colors[0],
                ms=3, label=r'Source points ($L = {}$)'.format(self.x.shape[0]))

        ax.plot(self.coord, self.d * np.ones_like(self.coord), 'o', color = colors[1],
                ms=3, label=r'Measurement points ($M = {}$)'.format(self.coord.shape[0]))

        # Vertical separation
        x_arrow = self.lb - 0.06 * span

        ax.annotate(
            '',
            xy=(x_arrow, self.d),
            xytext=(x_arrow, 0),
            arrowprops=dict(arrowstyle='<->', color='black')
        )

        ax.text(
            x_arrow - 0.02 * span,
            self.d / 2,
            r'$d$',
            ha='right',
            va='center',
            fontsize=13
        )

        # Labels
        ax.text(self.ub + 0.02 * span, 0,
                r'$x,\ \rho(x)$', va='center', fontsize=12)

        ax.text(self.ub + 0.02 * span, self.d,
                r"$x',\ g(x')$", va='center', fontsize=12)

        # Appearance
        ax.set_xlim(self.lb - margin, self.ub + 0.25 * span)
        ax.set_ylim(-0.25 * self.d, 1.25 * self.d)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(np.linspace(self.lb, self.ub, 5))
        ax.set_yticks(np.linspace(0, self.d, 5))
        ax.legend(loc='upper center', ncol=2, frameon=False)

        fig.tight_layout()

        return fig, ax

