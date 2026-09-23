r"""
Image deblurring as a discrete inverse problem.

This module provides a toy problem for reconstructing a grayscale image
from blurred and noisy observations. The image is degraded by convolution
with a Gaussian point-spread function (PSF) and additive Gaussian noise.

The forward problem can be implemented in three equivalent ways:

- Two-dimensional convolution using the fast Fourier transform (FFT).
- A separable formulation using two one-dimensional blur matrices.
- An explicit matrix formulation using the Kronecker product.

All three implementations use periodic boundary conditions and produce
the same noise-free blurred image, up to numerical precision.

The problem is intended for testing and illustrating regularization
methods for ill-conditioned linear inverse problems. The explicit
Kronecker representation is particularly useful for small images,
where standard matrix-based solvers and parameter-choice methods can
be applied directly.
"""

from importlib.resources import files
import numpy as np
import scipy
from PIL import Image
from pyregtools.utils import add_gaussian_noise

class ImageDeblur:
    r"""
    Two-dimensional image deblurring inverse problem.

    This class provides a discrete inverse problem for reconstructing
    a grayscale image from observations degraded by Gaussian blur and
    additive Gaussian noise.

    When ``image_name`` is specified, the reference image is loaded and
    preprocessed using :meth:`load_image`. The image is converted to
    grayscale, resized to ``shape``, and normalized to the interval
    [0, 1]. If ``image_name`` is None, a random reference image is
    generated instead.

    The forward problem can be evaluated using
    FFT-based convolution, separable matrix multiplication, or an
    explicit Kronecker product matrix.

    **Class attributes**

    .. list-table::
       :widths: 25 20 55
       :header-rows: 1

       * - Attribute
         - Type / shape
         - Description
       * - ``image_name``
         - str
         - Name of the reference image, or ``'None Image'`` for a
           randomly generated image.
       * - ``b_true``
         - ndarray, (M, L)
         - Reference image.
       * - ``shape``
         - tuple
         - Image dimensions ``(M, L)``.
       * - ``M``, ``L``
         - int
         - Number of rows and columns, respectively.
       * - ``n_pixels``
         - int
         - Total number of pixels, ``M * L``.
       * - ``psf``
         - ndarray
         - Normalized Gaussian point-spread function.
       * - ``sigma``
         - float
         - Standard deviation of the Gaussian PSF, in pixels.
       * - ``A_y``, ``A_x``
         - ndarray
         - Circulant blur matrices of shapes ``(M, M)`` and ``(L, L)``.
       * - ``A``
         - ndarray, (M*L, M*L)
         - Explicit Kronecker product matrix.
       * - ``H_fft``
         - ndarray, (M, L)
         - Fourier transform of the embedded convolution kernel.
       * - ``b_blur``
         - ndarray, (M, L)
         - Noise-free blurred image.
       * - ``b_noisy``
         - ndarray, (M, L)
         - Blurred image with additive Gaussian noise.

    Parameters
    ----------
    image_name : str or None, default="moon.png"
        Name of the reference image stored in the ``pyregtools.data``
        package. Supported formats include PNG and JPEG.

        If None, a random image is generated with independent samples
        from a standard normal distribution.

    shape : tuple of int or None, default=None
        Desired image shape ``(M, L)``, where ``M`` and ``L`` denote
        the number of rows and columns, respectively.

        If None, the default shape ``(16, 16)`` is used.

    Notes
    -----
    **Reference image**

    The reference image is represented by

    .. math::

        B_{\mathrm{true}} \in \mathbb{R}^{M \times L}.

    Its row-major vectorization is

    .. math::

        \mathbf{b}_{\mathrm{true}}
        =
        \operatorname{vec}_C(B_{\mathrm{true}})
        \in \mathbb{R}^{ML}.

    **Gaussian blurring**

    The image is degraded by convolution with a normalized Gaussian
    point-spread function (PSF). The PSF is separable, allowing the
    noise-free blurred image to be expressed as

    .. math::

        B_{\mathrm{blur}}
        =
        A_y B_{\mathrm{true}} A_x^T,

    where :math:`A_y` and :math:`A_x` are circulant matrices
    implementing one-dimensional Gaussian convolution along the
    vertical and horizontal image dimensions, respectively.

    Periodic boundary conditions are assumed.

    **Discrete inverse problem**

    Using row-major vectorization, the forward model becomes

    .. math::

        \mathbf{b}_{\mathrm{blur}}
        =
        A\mathbf{b}_{\mathrm{true}},

    with

    .. math::

        A = A_y \otimes A_x
        \in \mathbb{R}^{ML \times ML}.

    The noisy observations are modeled as

    .. math::

        \mathbf{b}_{\mathrm{noisy}}
        =
        A\mathbf{b}_{\mathrm{blur}}
        +
        \mathbf{n},

    where :math:`\mathbf{n}` denotes additive Gaussian noise.

    The inverse problem consists of recovering the reference image
    from the noisy observations and the blurring operator.

    **Forward-model implementations**

    Three equivalent methods are provided to compute the
    noise-free blurred image:

    - :meth:`blur_fft` applies circular convolution using the FFT.
    - :meth:`blur_separable` evaluates the separable matrix formulation.
    - :meth:`blur_kron` constructs and applies the explicit Kronecker
      product matrix.

    All three methods store the resulting image in ``b_blur``.

    The explicit Kronecker representation is intended for small
    images because the matrix dimensions grow with the total
    number of pixels.
    """

    def __init__(self, image_name="moon.png", shape=None):
        if shape is None:
            shape = (16,16)
        if image_name is None:
            self.image_name = 'None Image'
            self.b_true = np.random.normal(0, 1, size = shape)
        else:
            self.image_name = image_name
            self.b_true = self.load_image(image_name, shape)
        self.shape = self.b_true.shape
        self.M, self.L = self.shape
        self.n_pixels = self.M * self.L

    @staticmethod
    def load_image(image_name, shape=None):
        r"""
        Load and preprocess a grayscale image.

        The image is loaded from the ``pyregtools.data`` package,
        converted to grayscale, and represented as a floating-point
        array with values in the interval [0, 1].

        Parameters
        ----------
        image_name : str
            Name of the image file stored in ``pyregtools.data``.
            Supported formats include PNG and JPEG.
        shape : tuple of int or None, default=None
            Desired image shape ``(M, L)``, where ``M`` and ``L`` are
            the number of rows and columns, respectively. If None, the
            original image dimensions are preserved.

        Returns
        -------
        image : ndarray, shape (M, L)
            Grayscale image with floating-point values in the interval
            [0, 1].

        Notes
        -----
        If ``shape`` is specified, the image is resized using Lanczos
        resampling before conversion to a NumPy array.

        The image intensities are normalized according to

        .. math::

            B_{ij} = \frac{I_{ij}}{255},

        where :math:`I_{ij}` is the 8-bit grayscale intensity of pixel
        :math:`(i,j)`.
        """
        image_path = files("pyregtools.data").joinpath(image_name)
        with image_path.open("rb") as file:
            with Image.open(file) as img:
                img = img.convert("L")
                if shape is not None:
                    M, L = shape
                    img = img.resize((L, M), resample=Image.Resampling.LANCZOS)
                image = np.asarray(img, dtype=float) / 255.0
        return image

    def gaussian_psf(self, sigma=2.0):
        r"""
        Construct a normalized two-dimensional Gaussian point-spread function.

        Parameters
        ----------
        sigma : float, default=2.0
            Standard deviation of the Gaussian point-spread function (PSF),
            in pixels. Must be positive.

        Notes
        -----
        The Gaussian PSF is defined by

        .. math::

            h(x,y)
            =
            \frac{1}{2\pi\sigma^2}
            \exp\left(
                -\frac{x^2+y^2}{2\sigma^2}
            \right).

        The PSF is sampled on a square grid centered at the origin and
        extending approximately three standard deviations in each
        direction. The grid radius is

        .. math::

            r = \lceil 3\sigma \rceil,

        resulting in a discrete PSF of shape ``(2*r + 1, 2*r + 1)``.

        After sampling, the PSF is normalized such that

        .. math::

            \sum_{i,j} h_{ij} = 1.

        This normalization preserves the intensity of a constant image
        under convolution with periodic boundary conditions.

        The resulting PSF and its standard deviation are stored in
        ``psf`` and ``sigma``, respectively.

        Raises
        ------
        ValueError
            If ``sigma`` is not positive.
        """

        if sigma <= 0:
            raise ValueError("sigma must be positive.")

        radius = int(np.ceil(3 * sigma))

        coords = np.arange(-radius, radius + 1)

        X, Y = np.meshgrid(coords, coords)

        psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

        psf /= np.sum(psf)

        self.psf = psf
        self.sigma = sigma

    def blur_fft(self):
        r"""
        Apply Gaussian blur using FFT-based circular convolution.

        The reference image is convolved with the Gaussian point-spread
        function using the two-dimensional discrete Fourier transform.

        Returns
        -------
        b_blur : ndarray, shape (M, L)
            Noise-free blurred image.

        Notes
        -----
        The blurred image is computed as

        .. math::

            B_{\mathrm{blur}}
            =
            \mathcal{F}^{-1}
            \left\{
            H \odot \mathcal{F}(B_{\mathrm{true}})
            \right\},

        where :math:`\mathcal{F}` denotes the two-dimensional discrete
        Fourier transform, :math:`H` is the Fourier transform of the
        embedded PSF, and :math:`\odot` denotes element-wise
        multiplication.

        The PSF is embedded in an array of shape ``(M, L)`` and shifted
        so that its center corresponds to zero lag. Periodic boundary
        conditions are assumed.

        The Fourier-domain kernel and blurred image are stored in
        ``H_fft`` and ``b_blur``, respectively.

        Raises
        ------
        RuntimeError
            If ``gaussian_psf`` has not been called.
        ValueError
            If the PSF dimensions exceed the image dimensions.
        """
        if not hasattr(self, "psf"):
            raise RuntimeError("Call gaussian_psf() before blur_fft().")
        M, L = self.shape
        kh, kw = self.psf.shape
        if kh > M or kw > L:
            raise ValueError("The PSF must not exceed the image dimensions.")
        # Embed the PSF, placing its center at zero lag.
        kernel = np.zeros((M, L), dtype=float)
        cy, cx = kh // 2, kw // 2
        kernel[:kh, :kw] = self.psf
        # Move the PSF center to index (0, 0).
        kernel = np.roll(kernel, shift=(-cy, -cx), axis=(0, 1))
        # Fourier transform of the convolution kernel.
        self.H_fft = np.fft.fft2(kernel)
        # Apply circular convolution.
        self.b_blur = np.fft.ifft2(self.H_fft * np.fft.fft2(self.b_true)).real
        return self.b_blur

    def blur_separable(self):
        r"""
        Apply Gaussian blur using the separable matrix representation.

        The Gaussian PSF is separated into horizontal and vertical
        components and applied using two one-dimensional blur matrices.

        Returns
        -------
        b_blur : ndarray, shape (M, L)
            Noise-free blurred image.

        Notes
        -----
        The blurred image is computed as

        .. math::

            B_{\mathrm{blur}}
            =
            A_y B_{\mathrm{true}} A_x^T,

        where :math:`A_y \in \mathbb{R}^{M \times M}` and
        :math:`A_x \in \mathbb{R}^{L \times L}` are circulant matrices
        implementing Gaussian convolution along the vertical and
        horizontal image dimensions, respectively.

        The blur matrices are constructed internally by
        :meth:`build_blur_matrices`. Periodic boundary conditions are assumed.

        The resulting image is stored in ``b_blur``.

        Raises
        ------
        RuntimeError
            If :meth:`gaussian_psf` has not been called.
        ValueError
            If the PSF dimensions exceed the image dimensions.
        """
        self.build_blur_matrices()
        self.b_blur = self.A_y @ self.b_true @ self.A_x.T
        return self.b_blur

    def blur_kron(self):
        r"""
        Apply Gaussian blur using an explicit Kronecker product matrix.

        The two-dimensional blurring operator is constructed from the
        horizontal and vertical one-dimensional blur matrices using a
        Kronecker product.

        Returns
        -------
        b_blur : ndarray, shape (M, L)
            Noise-free blurred image.

        Notes
        -----
        Using row-major vectorization, the separable forward model

        .. math::

            B_{\mathrm{blur}}
            =
            A_y B_{\mathrm{true}} A_x^T

        can be written as

        .. math::

            \mathbf{b}_{\mathrm{blur}}
            =
            A\mathbf{b}_{\mathrm{true}},

        where

        .. math::

            A = A_y \otimes A_x.

        The reference image is vectorized using C-order (row-major)
        indexing before multiplication by :math:`A`. The resulting vector
        is subsequently reshaped to ``(M, L)``.

        The explicit forward matrix and blurred image are stored in
        ``A`` and ``b_blur``, respectively.

        The explicit Kronecker representation is intended for relatively
        small images because :math:`A` has shape ``(M*L, M*L)``.

        Raises
        ------
        RuntimeError
            If :meth:`gaussian_psf` has not been called.
        ValueError
            If the PSF dimensions exceed the image dimensions.
        """
        self.build_blur_matrices()
        self.A = np.kron(self.A_y, self.A_x)
        x = self.b_true.ravel(order="C")
        self.b_blur = (self.A @ x).reshape(self.shape, order="C")
        return self.b_blur

    def build_blur_matrices(self):
        r"""
        Construct the one-dimensional Gaussian blur matrices.

        The two-dimensional Gaussian point-spread function is separated
        into vertical and horizontal kernels, which are used to construct
        circulant blur matrices. The normalized one-dimensional kernels are 
        obtained from the two-dimensional PSF as

        .. math::

            g_y(i) = \sum_j h_{ij},
            \qquad
            g_x(j) = \sum_i h_{ij},

        where :math:`h_{ij}` denotes the discrete Gaussian PSF.

        The kernels are embedded into vectors of lengths ``M`` and ``L``
        and shifted so that zero lag corresponds to the first element.
        The resulting vectors define the first columns of the circulant
        matrices

        .. math::

            A_y \in \mathbb{R}^{M \times M},
            \qquad
            A_x \in \mathbb{R}^{L \times L}.

        These matrices implement one-dimensional circular convolution
        along the vertical and horizontal image dimensions, respectively.
        Consequently, the two-dimensional blur can be written as

        .. math::

            B_{\mathrm{blur}}
            =
            A_y B_{\mathrm{true}} A_x^T.

        Periodic boundary conditions are therefore assumed.

        The resulting matrices are stored in ``A_y`` and ``A_x``.

        Raises
        ------
        RuntimeError
            If :meth:`gaussian_psf` has not been called.
        ValueError
            If the PSF dimensions exceed the image dimensions.
        """

        if not hasattr(self, "psf"):
            raise RuntimeError("Call gaussian_psf() before build_blur_matrices().")
        #from scipy.linalg import circulant
        M, L = self.shape
        kh, kw = self.psf.shape
        if kh > M or kw > L:
            raise ValueError("The PSF must not exceed the image dimensions.")
        # Recover the normalized 1D Gaussian kernels.
        gy = self.psf.sum(axis=1)
        gx = self.psf.sum(axis=0)
        # Embed each kernel into a periodic sequence.
        cy = kh // 2
        cx = kw // 2
        hy = np.zeros(M)
        hx = np.zeros(L)
        hy[:kh] = gy
        hx[:kw] = gx
        # Place zero lag at index zero.
        hy = np.roll(hy, -cy)
        hx = np.roll(hx, -cx)
        # Each column represents the response to a unit impulse.
        self.A_y = scipy.linalg.circulant(hy)
        self.A_x = scipy.linalg.circulant(hx)

    def add_noise(self, snr = 25, seed = 0):
        r"""
        Add Gaussian noise to the blurred image.

        Gaussian noise is added to the noise-free blurred image according
        to a prescribed signal-to-noise ratio (SNR).

        Parameters
        ----------
        snr : float, default=25
            Target signal-to-noise ratio in decibels (dB).
        seed : int, default=0
            Seed used for reproducible noise generation.

        Notes
        -----
        The noisy observation is modeled as

        .. math::

            \mathbf{b}_{\mathrm{noisy}}
            =
            \mathbf{b}_{\mathrm{blur}} + \mathbf{n},

        where :math:`\mathbf{n}` is additive Gaussian noise.

        The noise-free blurred image is vectorized before noise is added
        using :func:`~pyregtools.utils.reg_utils.add_gaussian_noise`.
        The resulting noisy vector is then reshaped to the original image
        dimensions ``(M, L)`` and stored in ``b_noisy``.

        The noise-free blurred image ``b_blur`` must first be computed
        using one of ``blur_fft``, ``blur_separable``, or ``blur_kron``.

        Raises
        ------
        RuntimeError
            If the noise-free blurred image has not been computed.
        """
        if not hasattr(self, "b_blur"):
            raise RuntimeError("Compute the blurred image before calling add_noise().")
        b_blur_flat = self.b_blur.flatten()
        b_noisy_flat = add_gaussian_noise(b_blur_flat, snr = snr, seed = seed)
        self.b_noisy = np.reshape(b_noisy_flat, self.b_blur.shape)