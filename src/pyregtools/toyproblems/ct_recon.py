r"""
Two-dimensional parallel-beam computed tomography inverse problem.

This module provides a toy problem for reconstructing a two-dimensional
image from X-ray projection measurements. The image represents a
transverse slice of an object and is described by its spatially varying
linear attenuation coefficient.

For a monochromatic X-ray traveling along a path :math:`\ell_i`,
the transmitted intensity follows the Beer--Lambert law [1]_.

.. math::

    I_i
    =
    I_0
    \exp\left(
        -\int_{\ell_i}
        \mu(x,y)\,d\ell
    \right),

where :math:`I_0` is the incident intensity and :math:`\mu(x,y)` is
the linear attenuation coefficient. Taking the negative logarithm of
the normalized transmitted intensity gives

.. math::

    b_i
    =
    -\ln\left(\frac{I_i}{I_0}\right)
    =
    \int_{\ell_i}
    \mu(x,y)\,d\ell.

After discretizing the image into pixels with constant attenuation
coefficients, the projection measurements can be written as the
linear system

.. math::

    \mathbf{b}
    =
    A\mathbf{x},

where :math:`\mathbf{x}` contains the pixel attenuation coefficients
and each element :math:`A_{ij}` is the intersection length of the
:math:`i`-th X-ray with the :math:`j`-th pixel.

The sensing matrix is constructed using a two-dimensional
parallel-beam geometry. Projection measurements are acquired for
multiple ray offsets and projection angles and can be arranged as a
sinogram.

Poisson photon-counting noise can be introduced at the transmitted
intensity level before transforming the measurements back to the
logarithmic projection domain.

The problem is intended for illustrating image reconstruction and
regularization methods for ill-conditioned linear inverse problems.
The parallel-beam geometry is a simplified model of computed
tomography and does not represent the more complex acquisition
geometries used in modern clinical CT scanners.

References
----------
.. [1] Kak, A. C. and Slaney, A. C. *Principles of Computerized Tomographic Imaging*, SIAM, 2001.

"""

import numpy as np
import matplotlib.pyplot as plt

class CTReconstruction:
    """
    Two-dimensional parallel-beam computed tomography problem.

    This class generates a simplified computed tomography (CT) inverse
    problem using a two-dimensional parallel-beam acquisition geometry.
    The object is discretized into a rectangular grid of pixels with
    piecewise-constant attenuation coefficients.

    The main attributes created during initialization and subsequent
    method calls are:

    .. list-table::
       :widths: 20 20 60
       :header-rows: 1

       * - Attribute
         - Type
         - Description
       * - ``shape``
         - tuple of int
         - Number of pixels in the vertical and horizontal directions.
       * - ``size``
         - float
         - Side length of the square physical domain.
       * - ``dx``
         - float
         - Pixel width in the :math:`x` direction.
       * - ``dy``
         - float
         - Pixel height in the :math:`y` direction.
       * - ``x_edges``
         - ndarray
         - Coordinates of the pixel boundaries along the :math:`x`
           direction.
       * - ``y_edges``
         - ndarray
         - Coordinates of the pixel boundaries along the :math:`y`
           direction.
       * - ``angles``
         - ndarray
         - Projection angles in degrees.
       * - ``offsets``
         - ndarray
         - Perpendicular offsets of the parallel rays from the origin.
       * - ``A``
         - ndarray
         - Sensing matrix whose elements contain ray-pixel intersection
           lengths.
       * - ``x_true``
         - ndarray
         - Two-dimensional attenuation coefficient image representing
           the true phantom.
       * - ``b_true``
         - ndarray
         - Vector of noiseless projection measurements.
       * - ``sinogram_true``
         - ndarray
         - Noiseless projection measurements arranged as a sinogram.
       * - ``b_noisy``
         - ndarray
         - Vector of projection measurements obtained after introducing
           Poisson photon-counting noise.
       * - ``sinogram_noisy``
         - ndarray
         - Noisy projection measurements arranged as a sinogram.
    
    Parameters
    ----------
    shape : tuple of int, optional
        Number of pixels in the vertical and horizontal directions,
        respectively. Default is ``(16, 16)``.
    size : float, optional
        Side length of the square physical domain. The domain is centered
        at the origin and extends from ``-size / 2`` to ``size / 2`` in
        both spatial directions. Default is ``1.0``.
    """

    def __init__(self, shape=(16, 16), size=1.0):
        if len(shape) != 2:
            raise ValueError("shape must contain exactly two values.")
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("Image dimensions must be positive.")
        if size <= 0:
            raise ValueError("size must be positive.")

        self.shape = shape
        self.M, self.L = shape
        self.n_pixels = self.M * self.L
        self.size = size
        self.dx = size / self.L
        self.dy = size / self.M
        self.x_edges = np.linspace(-size / 2, size / 2, self.L + 1)
        self.y_edges = np.linspace(-size / 2, size / 2, self.M + 1)

    def sensing_mtx(self, angles, n_rays):
        r"""
        Build the sensing matrix for the parallel-beam CT problem.

        For a continuous attenuation coefficient :math:`\mu(x,y)`, a
        parallel-beam projection at angle :math:`\theta` and offset
        :math:`s` is given by the line integral

        .. math::

            b(s,\theta)
            =
            \int_{\ell(s,\theta)}
            \mu(x,y)\,d\ell,

        where the ray :math:`\ell(s,\theta)` satisfies

        .. math::

            x\cos\theta + y\sin\theta = s.

        The collection of these line integrals defines the Radon transform
        of :math:`\mu(x,y)`.

        Assuming a piecewise-constant attenuation coefficient within each
        image pixel, the continuous projection model is discretized as

        .. math::

            b_i
            =
            \sum_{j=1}^{N}
            A_{ij}x_j,

        or, in matrix form,

        .. math::

            \mathbf{b} = \mathbf{A}\mathbf{x},

        where :math:`x_j` is the attenuation coefficient of the
        :math:`j`-th pixel and :math:`A_{ij}` is the intersection length
        of the :math:`i`-th ray with the :math:`j`-th pixel.

        Parameters
        ----------
        angles : array_like
            One-dimensional array containing the projection angles in
            degrees.
        n_rays : int
            Number of parallel rays used at each projection angle.

        Notes
        -----
        The sensing matrix is stored in ``A`` with shape
        ``(n_angles * n_rays, n_pixels)``. The projection angles and ray
        offsets are stored in ``angles`` and ``offsets``, respectively.

        Rays are ordered first by projection angle and then by detector
        offset. Therefore, each consecutive block of ``n_rays`` rows of
        ``A`` corresponds to one projection angle.

        The detector offsets cover the range required for all projection
        angles of the square image domain. Rays are placed at the centers
        of equally spaced detector intervals, avoiding rays located exactly
        at the limiting offsets of the domain.
        """
        if n_rays < 2:
            raise ValueError("The number of rays must be at least 2.")
        self.angles = np.asarray(angles, dtype=float)

        if self.angles.ndim != 1:
            raise ValueError("angles must be a one-dimensional array.")

        max_offset = self.size / np.sqrt(2)
        offset_edges = np.linspace(-max_offset, max_offset, n_rays + 1)
        self.offsets = 0.5 * (offset_edges[:-1] + offset_edges[1:])
        n_meas = len(self.angles) * n_rays
        self.A = np.zeros((n_meas, self.n_pixels))
        row = 0
        for angle in self.angles:
            for offset in self.offsets:
                self.A[row, :] = self._ray_pixel_lengths(theta_deg=angle,offset=offset)
                row += 1

    def _ray_pixel_lengths(self, theta_deg, offset):
        r"""
        Compute the intersection length of a ray with every pixel.

        The ray is defined in normal form as

        .. math::

            x\cos\theta + y\sin\theta = s,

        where :math:`\theta` is the projection angle and :math:`s` is the
        perpendicular offset of the ray from the origin. A parametric
        representation of the ray is

        .. math::

            x(\lambda) &= s\cos\theta - \lambda\sin\theta, \\
            y(\lambda) &= s\sin\theta + \lambda\cos\theta.

        The intersections of the ray with the vertical and horizontal
        pixel boundaries are computed first. Consecutive intersection
        points define segments lying inside individual pixels. The length
        of each segment is assigned to the corresponding pixel.

        Parameters
        ----------
        theta_deg : float
            Projection angle in degrees.
        offset : float
            Perpendicular offset :math:`s` of the ray from the origin.

        Returns
        -------
        lengths : ndarray, shape (n_pixels,)
            Intersection length of the ray with each pixel, flattened in
            row-major (C) order. Pixels not intersected by the ray have
            zero length.

        Notes
        -----
        Since the direction vector

        .. math::

            (-\sin\theta,\cos\theta)

        has unit norm, differences in the ray parameter :math:`\lambda`
        correspond directly to physical path lengths.

        With the angle convention used here, :math:`\theta=0^\circ`
        corresponds to the vertical ray :math:`x=s`, while
        :math:`\theta=90^\circ` corresponds to the horizontal ray
        :math:`y=s`.
        """
        theta = np.deg2rad(theta_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        lambdas = []
        # Intersections with vertical grid lines.
        if not np.isclose(s, 0.0):
            lam_x = (offset * c - self.x_edges) / s
            y = offset * s + lam_x * c
            valid = ((y >= self.y_edges[0]) & (y <= self.y_edges[-1]))
            lambdas.extend(lam_x[valid])

        # Intersections with horizontal grid lines.
        if not np.isclose(c, 0.0):
            lam_y = (self.y_edges - offset * s) / c
            x = offset * c - lam_y * s
            valid = ((x >= self.x_edges[0]) & (x <= self.x_edges[-1]))
            lambdas.extend(lam_y[valid])

        if len(lambdas) < 2:
            return np.zeros(self.n_pixels)

        lambdas = np.unique(np.asarray(lambdas))
        lambdas.sort()
        lengths = np.zeros(self.shape)

        for k in range(len(lambdas) - 1):

            lam_1 = lambdas[k]
            lam_2 = lambdas[k + 1]
            lam_mid = 0.5 * (lam_1 + lam_2)
            x_mid = offset * c - lam_mid * s
            y_mid = offset * s + lam_mid * c

            col = np.searchsorted(self.x_edges, x_mid, side="right") - 1
            row = np.searchsorted(self.y_edges, y_mid, side="right") - 1
            if 0 <= row < self.M and 0 <= col < self.L:
                lengths[row, col] += lam_2 - lam_1
        return lengths.ravel(order="C")

    def create_phantom(self):
        r"""
        Create the true attenuation coefficient image.

        A simple two-dimensional phantom is constructed on the pixel grid.
        The phantom consists of a main circular object containing two
        smaller circular inclusions with different attenuation coefficients.

        The main object has an attenuation coefficient of ``1.0``. The two
        inclusions have attenuation coefficients of ``1.5`` and ``0.5``,
        respectively. Pixels outside the main object have zero attenuation.

        Notes
        -----
        The phantom is evaluated at the pixel centers and stored in
        ``x_true`` as a two-dimensional array with shape ``shape``.

        The attenuation coefficients are dimensionless in this toy problem.
        Their values are chosen to provide spatial contrast rather than to
        represent a particular material or tissue.
        """
        x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        X, Y = np.meshgrid(x_centers, y_centers)
        self.x_true = np.zeros(self.shape)
        # Main circular object
        mask = X**2 + Y**2 <= (0.35 * self.size)**2
        self.x_true[mask] = 1.0
        # Smaller inclusion with larger attenuation
        mask = ((X - 0.12 * self.size)**2 + (Y - 0.10 * self.size)**2 <= (0.10 * self.size)**2)
        self.x_true[mask] = 1.5
        # Smaller inclusion with lower attenuation
        mask = ((X + 0.15 * self.size)**2 + (Y + 0.12 * self.size)**2 <= (0.08 * self.size)**2)
        self.x_true[mask] = 0.5

    def noiseless_meas(self, ):
        r"""
        Compute the noiseless CT projection measurements.

        The noiseless measurements are obtained from the discrete forward
        model

        .. math::

            \mathbf{b}_{\mathrm{true}}
            =
            \mathbf{A}\mathbf{x}_{\mathrm{true}},

        where :math:`\mathbf{A}` is the sensing matrix and
        :math:`\mathbf{x}_{\mathrm{true}}` is the vectorized attenuation
        coefficient image.

        Notes
        -----
        The measurement vector is stored in ``b_true`` with shape
        ``(n_angles * n_rays,)``.

        The measurements are also arranged as a two-dimensional sinogram
        and stored in ``sinogram_true`` with shape
        ``(n_angles, n_rays)``.

        The sensing matrix and true phantom must be generated with
        :meth:`sensing_mtx()` and :meth:`create_phantom()`, respectively, before
        calling this method.
        """
        if not hasattr(self, "A"):
            raise RuntimeError("Call sensing_mtx() before noiseless_meas().")

        if not hasattr(self, "x_true"):
            raise RuntimeError("Call create_phantom() before noiseless_meas().")
        self.b_true = self.A @ self.x_true.ravel()
        self.sinogram_true = self.b_true.reshape(len(self.angles), len(self.offsets))

    def add_noise(self, photon_count=1e5, seed=0):
        r"""
        Add Poisson photon-counting noise to the CT measurements.

        The expected number of transmitted photons for the :math:`i`-th
        projection measurement is modeled as

        .. math::

            \lambda_i
            =
            I_0 \exp(-b_{i,\mathrm{true}}),

        where :math:`I_0` is the incident photon count and
        :math:`b_{i,\mathrm{true}}` is the noiseless line integral.
        The detected photon count is sampled according to

        .. math::

            N_i \sim \operatorname{Poisson}(\lambda_i).

        The noisy projection measurement is then obtained using the
        logarithmic transformation

        .. math::

            b_{i,\mathrm{noisy}}
            =
            -\ln\left(\frac{N_i}{I_0}\right).

        Parameters
        ----------
        photon_count : float, default=1e5
            Incident photon count :math:`I_0` used for each ray.
            Larger values produce smaller relative photon-counting
            fluctuations.
        seed : int, , default=0
            Seed used to initialize the random number generator.

        Notes
        -----
        The noisy projection vector is stored in ``b_noisy``. The same
        measurements are arranged as a two-dimensional sinogram and stored
        in ``sinogram_noisy`` with shape ``(n_angles, n_rays)``.

        Photon counts equal to zero are replaced by one before applying the
        logarithm to avoid undefined values.

        The resulting noise in the logarithmic projection domain is
        signal-dependent and is not, in general, additive white Gaussian
        noise.

        :meth:`noiseless_meas()` must be called before this method.
        """
        if not hasattr(self, "b_true"):
            raise RuntimeError("Call noiseless_meas() before add_noise().")

        if photon_count <= 0:
            raise ValueError("photon_count must be positive.")

        rng = np.random.default_rng(seed)
        expected_counts = (photon_count * np.exp(-self.b_true))
        counts = rng.poisson(expected_counts)
        counts = np.maximum(counts, 1)
        self.b_noisy = -np.log(counts / photon_count)
        self.sinogram_noisy = self.b_noisy.reshape(len(self.angles), len(self.offsets))

    def plot_true_phantom(self):
        """
        Plot the true attenuation coefficient image.

        Notes
        -----
        :meth:`create_phantom()` must be called before this method.
        """
        plt.figure(figsize = (6,4))
        plt.imshow(self.x_true, origin="lower", cmap='Greys',
                   extent=[-self.size / 2, self.size / 2, -self.size / 2, self.size / 2])
        plt.colorbar(label="Attenuation coefficient")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Image shape: {}".format(self.x_true.shape))

    def plot_true_sinogram(self):
        """
        Plot the noiseless CT sinogram.

        Notes
        -----
        :meth:`noiseless_meas()` must be called before this method.
        """
        plt.figure(figsize = (6,4))
        plt.imshow(self.sinogram_true, aspect="auto", origin="lower", cmap='Greys',
                    extent=[self.offsets[0], self.offsets[-1], self.angles[0], self.angles[-1]])
        plt.colorbar(label="Projection")
        plt.xlabel("Detector offset")
        plt.ylabel("Projection angle [deg]")
        plt.title("Sinogram shape: {}".format(self.sinogram_true.shape))

    def plot_noisy_sinogram(self):
        """
        Plot the noisy CT sinogram.

        Notes
        -----
        :meth:`add_noise()` must be called before this method.
        """
        plt.figure(figsize = (6,4))
        plt.imshow(self.sinogram_noisy, aspect="auto", origin="lower", cmap='Greys',
                    extent=[self.offsets[0], self.offsets[-1], self.angles[0], self.angles[-1]])
        plt.colorbar(label="Projection")
        plt.xlabel("Detector offset")
        plt.ylabel("Projection angle [deg]")
        plt.title("Noisy sinogram shape: {}".format(self.sinogram_noisy.shape))

