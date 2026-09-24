r"""
Multiple exponential decay as a discrete inverse problem.

This module provides a toy problem for reconstructing the decay
components of an acoustical energy decay curve (EDC). The EDC is
modeled as a superposition of exponential decays with different
decay times and an optional integrated noise term [1]_, [2]_.

The energy decay model is

.. math::

    b(t)
    =
    \sum_{j=1}^{N_d}
    a_j
    \exp\left(
        -\frac{6\ln(10)}{T_j}t
    \right)
    +
    a_n(t_{\max}-t),

where :math:`N_d` is the number of decay components,
:math:`a_j` and :math:`T_j` are the amplitude and 60-dB energy
decay time of the :math:`j`-th component, respectively, and
:math:`a_n` is the amplitude of the integrated noise term.

The inverse problem is formulated by defining a set of candidate decay
times and constructing a sensing matrix whose columns contain the
corresponding exponential decay functions. The amplitudes of these
components can then be estimated from noisy observations of the EDC.

The problem is intended for illustrating regularization methods for
ill-conditioned linear inverse problems, particularly when the columns
of the sensing matrix have substantially different norms.

References
----------
.. [1] Xiang, X. and Goggans, P. M., *Evaluation of decay times in coupled spaces: Bayesian
    parameter estimation*, The Journal of the Acoustical Society of America, 110(3), 
    2001, p.1415--1424.

.. [2] Jasa, T. and Xiang, X., *Nested sampling applied in Bayesian room-acoustics decay
    analysis*, The Journal of the Acoustical Society of America, 132(5), 
    2012, p.3251--3262.
"""

import numpy as np
import matplotlib.pyplot as plt

class MultiDecay:
    r"""
    Multiple exponential decay inverse problem.

    This class provides a discrete inverse problem for reconstructing
    the components of an acoustical energy decay curve (EDC). The EDC
    is modeled as a superposition of exponential decays with different
    60-dB decay times and an optional integrated noise term.

    A sensing matrix can be constructed from a set of candidate decay
    times, allowing the amplitudes of the corresponding decay components
    to be estimated from noisy observations.

    **Energy decay model**
    
    The noise-free energy decay curve is modeled as

    .. math::

        b_{\mathrm{true}}(t)
        =
        \sum_{j=1}^{N_d}
        a_j
        \exp\left(
            -\frac{6\ln(10)}{T_j}t
        \right)
        +
        a_n(t_{\max}-t),

    where :math:`N_d` is the number of exponential components,
    :math:`a_j` is the amplitude of the :math:`j`-th component,
    :math:`T_j` is its 60-dB energy decay time, and :math:`a_n`
    is the amplitude of the integrated noise term.

    For an individual exponential component,

    .. math::

        10\log_{10}
        \left[
            \frac{b_j(T_j)}{b_j(0)}
        \right]
        =
        -60\ \mathrm{dB},

    which defines :math:`T_j` as the time required for the energy
    to decay by 60 dB.

    **Discrete inverse problem**

    A set of :math:`L` candidate decay times
    :math:`\widetilde{T}_j` is used to construct the sensing matrix

    .. math::

        A_{ij}
        =
        \exp\left(
            -\frac{6\ln(10)}{\widetilde{T}_j}t_i
        \right),

    where :math:`t_i` denotes the :math:`i`-th measurement time.

    The corresponding discrete forward model is

    .. math::

        \mathbf{b}
        =
        A\mathbf{x},

    where the entries of :math:`\mathbf{x}` represent the amplitudes
    associated with the candidate decay times.

    Short decay times produce rapidly decreasing exponential functions
    and consequently columns of relatively small norm. Longer decay
    times produce more slowly decreasing functions and columns with
    larger norms. The problem therefore provides an example in which
    the columns of the sensing matrix can have substantially different
    scales.

    If the integrated noise term is included in the sensing matrix,
    the additional column is

    .. math::

        A_{i,L+1} = t_{\max} - t_i.

    The inverse problem consists of estimating the decay amplitudes
    from noisy observations of the energy decay curve.    
    
    .. list-table::
        :widths: 25 20 55
        :header-rows: 1

        * - Attribute
          - Type / shape
          - Description
        * - ``fs``
          - int
          - Sampling rate of the energy decay curve, in samples per
            second.
        * - ``time``
          - ndarray, (M,)
          - Time vector of the energy decay curve, in seconds.
        * - ``t_decays``
          - array-like, (N_d,)
          - 60-dB decay times used to generate the true EDC.
        * - ``amps``
          - array-like, (N_d,)
          - Amplitudes of the exponential components used to generate
            the true EDC.
        * - ``amp_int_noise``
          - float
          - Amplitude of the integrated noise term.
        * - ``b_true``
          - ndarray, (M,)
          - Noise-free energy decay curve.
        * - ``b_noisy``
          - ndarray, (M,)
          - Energy decay curve with random perturbations.
        * - ``t_decs``
          - ndarray, (L,)
          - Candidate 60-dB decay times used to construct the sensing
            matrix.
        * - ``A``
          - ndarray, (M, L) or (M, L + 1)
          - Sensing matrix containing the candidate exponential decay
            functions. When the integrated noise term is included, an
            additional column is appended.

    Parameters
    ----------
    t_max : float, default=5
        Maximum measurement time, in seconds. Must be positive.
    fs : int, default=1000
        Sampling rate of the energy decay curve, in samples per second.
        Must be at least 10.
    """
    def __init__(self, t_max = 5, fs = 1000):
        if t_max <= 0:
            raise ValueError("Maximum measurement time must be positive.")
        if fs < 10:
            raise ValueError("Sample rate must be at least 10 samples per second.")
        self.fs = fs   
        self.time = np.arange(1/fs, t_max+1/fs, 1/fs) # time vector of measurement
    
    def single_decay(self, amp = 1, t_decay = 1):
        r"""
        Compute a single exponential energy decay.

        Parameters
        ----------
        amp : float, default=1
            Amplitude of the exponential decay. Must be non-negative
            and real.
        t_decay : float, default=1
            60-dB energy decay time, in seconds. Must be positive
            and real.

        Returns
        -------
        exp_decay : ndarray, shape (M,)
            Exponential energy decay evaluated at the measurement times.

        Notes
        -----
        The energy decay is defined as

        .. math::

            b(t)
            =
            a
            \exp\left(
                -\frac{6\ln(10)}{T}t
            \right),

        where :math:`a` is the decay amplitude and :math:`T` is the
        60-dB energy decay time.

        At :math:`t=T`, the energy has decreased by 60 dB relative to
        its value at :math:`t=0`:

        .. math::

            10\log_{10}
            \left[
                \frac{b(T)}{b(0)}
            \right]
            =
            -60\ \mathrm{dB}.

        Raises
        ------
        ValueError
            If ``amp`` is negative or not real.
        ValueError
            If ``t_decay`` is not positive or not real.
        """
        if amp <= 0 or not np.isreal(amp):
            raise ValueError("Decay amplitude must be non-negative and real.")
        if t_decay <=0 or not np.isreal(t_decay):
            raise ValueError("Decay time must be non-negative and real.")
        decay_rate = 6 * np.log(10) / t_decay
        exp_decay = amp*np.exp(-decay_rate*self.time)
        return exp_decay
        
    def integrated_noise_term(self, amp_int_noise):
        r"""
        Compute the integrated noise term of the energy decay curve.

        Parameters
        ----------
        amp_int_noise : float
            Amplitude of the integrated noise term. Must be non-negative
            and real.

        Returns
        -------
        int_noise : ndarray, shape (M,)
            Integrated noise term evaluated at the measurement times.

        Notes
        -----
        The integrated noise term is modeled as

        .. math::

            b_n(t)
            =
            a_n (t_{\max} - t),

        where :math:`a_n` is the amplitude of the term and
        :math:`t_{\max}` is the maximum measurement time.

        This term represents the contribution of stationary background
        noise after backward integration of the squared impulse response.
        It decreases linearly with time and is distinct from the random
        perturbations added by :meth:`add_noise`.

        Raises
        ------
        ValueError
            If ``amp_int_noise`` is negative or not real.
        """
        if amp_int_noise < 0 or not np.isreal(amp_int_noise):
            raise ValueError("Integrated noise amplitude must be larger or equal than 0 and real.")
        int_noise = amp_int_noise*(self.time[-1]-self.time)
        return int_noise
    
    def compose_decay(self, t_decays = (0.5, 1.00), amps = (1.0, 0.25), amp_int_noise = 0):
        r"""
        Compose a multiple exponential energy decay curve.

        The noise-free energy decay curve is constructed by summing
        exponential components with specified decay times and amplitudes,
        together with an optional integrated noise term.

        Parameters
        ----------
        t_decays : array-like, default=(0.5, 1.0)
            60-dB energy decay times, in seconds. Each value must be
            positive and real.
        amps : array-like, default=(1.0, 0.25)
            Amplitudes of the exponential decay components. Must have
            the same length as ``t_decays``.
        amp_int_noise : float, default=0
            Amplitude of the integrated noise term. Must be non-negative
            and real.

        Notes
        -----
        The noise-free energy decay curve is modeled as

        .. math::

            b_{\mathrm{true}}(t)
            =
            \sum_{j=1}^{N_d}
            a_j
            \exp\left(
                -\frac{6\ln(10)}{T_j}t
            \right)
            +
            a_n(t_{\max}-t),

        where :math:`N_d` is the number of exponential components,
        :math:`a_j` and :math:`T_j` are the amplitude and 60-dB energy
        decay time of the :math:`j`-th component, respectively, and
        :math:`a_n` is the amplitude of the integrated noise term.

        The decay times, amplitudes, and integrated noise amplitude are
        stored in ``t_decays``, ``amps``, and ``amp_int_noise``,
        respectively. The resulting noise-free energy decay curve is
        stored in ``b_true``.

        Raises
        ------
        ValueError
            If ``t_decays`` and ``amps`` do not have the same length.
        ValueError
            If an invalid decay time, decay amplitude, or integrated
            noise amplitude is supplied.
        """
        if len(t_decays) != len(amps):
            raise ValueError(f"Decay and amplitude arrays must have the same length. Got {len(t_decays)} and {len(amps)}")
        self.t_decays = t_decays
        self.amps = amps
        self.amp_int_noise = amp_int_noise
        self.b_true = np.zeros(len(self.time))
        for jd, td in enumerate(t_decays):
            self.b_true += self.single_decay(amp = amps[jd], t_decay = td)
        self.b_true += self.integrated_noise_term(amp_int_noise = amp_int_noise)
        
    def add_noise(self, snr = 30, seed = 0):
        r"""
        Add signal-dependent Gaussian noise to the energy decay curve.

        Gaussian random perturbations are added to the noise-free energy
        decay curve, with a standard deviation proportional to the
        instantaneous decay amplitude.

        Parameters
        ----------
        snr : float, default=30
            Signal-to-noise ratio controlling the relative standard
            deviation of the perturbations, in decibels (dB).
        seed : int, default=0
            Seed used for reproducible noise generation.

        Notes
        -----
        The noisy energy decay curve is modeled as

        .. math::

            b_{\mathrm{noisy}}(t_i)
            =
            b_{\mathrm{true}}(t_i) + n_i,

        where the noise samples are independently drawn as

        .. math::

            n_i
            \sim
            \mathcal{N}
            \left(
                0,\,
                \sigma_i^2
            \right),

        with

        .. math::

            \sigma_i
            =
            10^{-\mathrm{SNR}/20}
            b_{\mathrm{true}}(t_i).

        Thus, the standard deviation of the perturbation decreases with
        the instantaneous amplitude of the energy decay curve. This
        signal-dependent noise model is consistent with energy decay
        curves obtained from backward integration of squared impulse
        responses.

        The resulting noisy energy decay curve is stored in ``b_noisy``.

        Raises
        ------
        RuntimeError
            If the noise-free energy decay curve has not been computed.
        """
        if not hasattr(self, "b_true"):
            raise RuntimeError("Call compose_decay() before add_noise().")
        np.random.seed(seed)
        noise_amp = 10**(-snr/20)
        n = np.random.normal(loc = 0.0, scale = noise_amp*self.b_true, size = len(self.b_true))
        self.b_noisy = self.b_true + n
    
    def sensing_mtx(self, time_decay_range = (0.05, 3), num_cand_decay = 30,
                    include_int_noise = False):
        r"""
        Construct the sensing matrix for the multiple-decay problem.

        The sensing matrix is constructed from a set of candidate 60-dB
        energy decay times. Each column contains a unit-amplitude
        exponential decay evaluated at the measurement times.

        Parameters
        ----------
        time_decay_range : tuple of float, default=(0.05, 3)
            Minimum and maximum candidate 60-dB energy decay times, in
            seconds.
        num_cand_decay : int, default=30
            Number of candidate decay times. Must be at least 2.
        include_int_noise : bool, default=False
            If True, append a column representing the integrated noise
            term.

        Notes
        -----
        The candidate decay times are uniformly distributed between the
        limits specified by ``time_decay_range``:

        .. math::

            \widetilde{T}_j
            \in
            [T_{\min}, T_{\max}],
            \qquad j=1,\ldots,L,

        where :math:`L` is ``num_cand_decay``.

        The entries of the sensing matrix are

        .. math::

            A_{ij}
            =
            \exp\left(
                -\frac{6\ln(10)}{\widetilde{T}_j}t_i
            \right),

        where :math:`t_i` is the :math:`i`-th measurement time and
        :math:`\widetilde{T}_j` is the :math:`j`-th candidate 60-dB
        decay time.

        The resulting discrete forward model is

        .. math::

            \mathbf{b} = A\mathbf{x},

        where the entries of :math:`\mathbf{x}` represent the amplitudes
        associated with the candidate decay times.

        If ``include_int_noise`` is True, an additional column is appended
        to the sensing matrix according to

        .. math::

            A_{i,L+1} = t_{\max} - t_i.

        In this case, the corresponding last entry of :math:`\mathbf{x}`
        represents the amplitude of the integrated noise term.

        Short candidate decay times produce rapidly decreasing
        exponentials and columns with relatively small norms, whereas
        longer decay times produce more slowly decreasing exponentials
        and columns with larger norms.

        The candidate decay times and sensing matrix are stored in
        ``t_decs`` and ``A``, respectively.

        Raises
        ------
        ValueError
            If ``time_decay_range`` does not contain exactly two values.
        ValueError
            If ``num_cand_decay`` is smaller than 2.
        """
        if len(time_decay_range) != 2:
            raise ValueError(f"The shape of time_decay_range must be 2. Got {len(time_decay_range)}.")
        if time_decay_range[0] <= 0 or time_decay_range[1] <= 0:
            raise ValueError("Candidate decay times must be positive.")
        if time_decay_range[0] >= time_decay_range[1]:
            raise ValueError("The upper limit of time_decay_range must be larger "
                "than the lower limit.")
        if num_cand_decay < 2:
            raise ValueError("The number of candidate decays must be at least 2.")
        self.t_decs = np.linspace(time_decay_range[0], time_decay_range[1], num_cand_decay)
        self.A = np.zeros((len(self.time), len(self.t_decs)))
        for jd, td in enumerate(self.t_decs):
            self.A[:, jd] = self.single_decay(amp = 1, t_decay = td)
        if include_int_noise:
            self.A = np.hstack((self.A, np.array([(self.time[-1]-self.time)]).T))
    
    def plot_edc(self):
        r"""
        Plot the energy decay curve and its components.

        The noise-free and noisy energy decay curves are plotted in
        decibels together with the individual exponential decay
        components. If present, the integrated noise term is also shown.

        Notes
        -----
        The noise-free and noisy energy decay curves are independently
        normalized by their respective maximum values and expressed in
        decibels as

        .. math::

            L(t)
            =
            10\log_{10}
            \left(
                \frac{b(t)}{\max (b(t))}
            \right).

        The individual exponential components and the integrated noise
        term are not normalized before conversion to decibels. Their
        original relative amplitudes are therefore preserved, allowing
        the contribution of each term along the energy decay curve to
        be visualized.

        The displayed range is limited to -60 dB to 2 dB.
        """
        plt.figure(figsize = (6,4))
        plt.plot(self.time, 10*np.log10(self.b_true/np.amax(self.b_true)), '--', 
                 color = 'grey', linewidth = 1.5, label = "True EDC")
        plt.plot(self.time, 10*np.log10(self.b_noisy/np.amax(self.b_noisy)), '-', 
                color = 'k', linewidth = 1.0, label = "Noisy EDC")
        for jd, td in enumerate(self.t_decays):
            exp_decay = self.single_decay(amp = self.amps[jd], t_decay = td)
            plt.plot(self.time, 10*np.log10(exp_decay), ':', label = "Decay {} [s]".format(td))
        if self.amp_int_noise != 0:
            dec_int_noise = self.integrated_noise_term(amp_int_noise = self.amp_int_noise)
            plt.plot(self.time[:-2], 10*np.log10(dec_int_noise[:-2]), ':', 
                    color = 'grey', label = "Integrated noise term")
        plt.legend()
        plt.grid(linestyle = '--')
        plt.xlim([0, self.time[-1]])
        plt.ylim([-60, 2])
        plt.xlabel("Time [s]")
        plt.ylabel("EDC [dB]")
        plt.tight_layout()