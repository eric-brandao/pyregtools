""" The package contains some useful methods to analyze inverse problems, including.

- plotting the column vectors of matrices (useful to show how SVD forms a spectral basis),
- execute the Picard plot (gives an overview of useful singular values),
- error metrics such as NMSE and MAE.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_colvecs(U, rows = 4, cols = 4, figsize = (8,5), ylim = (-0.2,0.2)):
    """ Plot the column vectors of a matrix.

    It is valid for any real matrix. If the matrix is complex, the real part is used.
    The method is particularly useful to demonstrate how the left or right singular vector
    matrices form a spectral basis for the forward problem and how it relates to the faster
    decay of high frequency components present in the measured signal.

    The first ``rows*cols`` will be plotted.

    Parameters
    ----------
        U : ndarray
            matrix which will be plotted.
        rows : int
            number of subfigure rows in the figure.
        cols : int
            number of subfigure columns in the figure.
        figsize : tuple
            figure size.
        ylim : tuple
            min, max value of the y axis.
    """
    # casting
    U = np.real(U)
    # figure
    fig, axs = plt.subplots(rows, cols, figsize = figsize, sharex=True, sharey=True)
    j = 0
    for row in np.arange(rows):
        for col in np.arange(cols):
            axs[row,col].plot(U[:,j], 'k', alpha = 0.8, label = r"{}".format(j))
            axs[row,col].legend()
            axs[row,col].set_xlim((0,len(U[:,j])))
            axs[row,col].set_ylim(ylim)
            axs[row,col].grid(linestyle = '--')
            j += 1
            axs[rows-1,col].set_xlabel(r'$i$')            
        axs[row,0].set_ylabel(r'col. vectors')
    plt.tight_layout()
    
def plot_picard(u,s,b, noise_norm = None, figsize = (5,4)):
    """ Make the Picard plot.

    The Picard plot is useful to analyze how the SVD of the
    sensing matrix :math:`\\mathbf{A} = \\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}^{H}`
    relates to the measured data :math:`\\mathbf{b}`.

    The Picard plot compares the following quantities:

    - the decaying singular values: :math:`\\sigma_i`.
    - the absolute values in: :math:`|\\textbf{u}_{i}^{H}\\textbf{b}|`.
    - the absolute values in: :math:`|\\textbf{u}_{i}^{H}\\textbf{b}| / \\sigma_i`.

    One useful intuition is that the inner product between :math:`\\textbf{b}` and 
    :math:`\\textbf{u}_{i}` measures size of the projection of the data in the direction
    of the i-th column vector of :math:`\\mathbf{U}`.

    Parameters
    ----------
        u : ndarray
            Left singular vectors.
        s : ndarray, shape (:math:`\\text{min}(M,L)`)
            Singular values in descending order.
        b : ndarray, shape (M,)
            Measurement vector.
        noise_norm : float or None
            Estimated noise norm, if available.
        figsize : tuple
            Figure size as ``(width, height)`` in inches.
    """
    # condition number
    cond_number = s[0]/s[-1]    
    # beta
    beta = np.abs(np.conj(u.T) @ b)    
    # Figure
    plt.figure(figsize = figsize)
    plt.semilogy(np.abs(s), '+k', label = r'$\sigma$')
    plt.semilogy(beta, 'xr', label = r'$|U^T b|$')
    plt.semilogy(beta/s, '.b', label = r'$|U^T b|/\sigma$')
    plt.semilogy(np.finfo(float).eps*s[0]*np.ones(len(s)), '--', linewidth = 2, 
                 color = 'Grey', label = r'eps$\cdot \sigma_1$')
    if noise_norm is not None:
        plt.semilogy((noise_norm/np.linalg.norm(b))*s[0]*np.ones(len(s)), '--', linewidth = 2, 
                     color = 'Grey', 
                     label = r'$\left\|n\right\|_2/\left\|b\right\|_2 \cdot \sigma_1$')
    plt.legend(loc = 'lower left')
    minval = np.amin([0.1*s[-1], 0.1*np.finfo(float).eps*s[0]])
    plt.ylim((minval, 100*s[0]))
    plt.xlabel(r'$i$')
    plt.ylabel(r'$\sigma_i$, $|U^Tb|$, $|U^Tb|/\sigma_i$')
    plt.title('cond(A) = {0:.2f}'.format(cond_number), loc='right')
    plt.grid()
    plt.tight_layout()
    
def nmse(x_meas, x_ref):
    """ Computes the normalized mean squared error (NMSE).

    The NMSE is given by

    .. math::
        
            \\text{NMSE} = \\frac{\\|\\textbf{x}_{\\text{meas}}-\\textbf{x}_{\\text{ref}}\\|_{2}^{2}}{\\|\\textbf{x}_{\\text{ref}}\\|_{2}^{2}},

    where :math:`\\textbf{x}_{\\text{meas}}` is a vector with a measured quantity 
    (it can be a computed solution or a reconstructed quantity of interest), and 
    :math:`\\textbf{x}_{\\text{ref}}` is a vector with a reference value for the 
    measured quantity.

    Parameters
    ----------
        x_meas : ndarray, shape (N,)
            A measured quantity representing a given solution or a reconstruction of
            a quantity of interest.
        x_ref : ndarray, shape (N,)
            Reference value for the measured quantity.
    
    Returns
    -------
        nnse : float
            Estimated NMSE.
    """
    nmse = (np.linalg.norm(x_meas-x_ref)/np.linalg.norm(x_ref))**2
    return nmse

def mae(x_meas, x_ref):
    """ Computes the mean absolute error (MAE).

    The MAE is given by

    .. math::
        
            \\text{MAE} = \\frac{1}{N}\\|\\textbf{x}_{\\text{meas}}-\\textbf{x}_{\\text{ref}}\\|_{2},

    where :math:`\\textbf{x}_{\\text{meas}}` is a vector with a measured quantity 
    (it can be a computed solution or a reconstructed quantity of interest), and 
    :math:`\\textbf{x}_{\\text{ref}}` is a vector with a reference value for the 
    measured quantity. Both vectors have :math:`N` components.

    Parameters
    ----------
        x_meas : ndarray, shape (N,)
            A measured quantity representing a given solution or a reconstruction of
            a quantity of interest.
        x_ref : ndarray, shape (N,)
            Reference value for the measured quantity.
    
    Returns
    -------
        mae : float
            Estimated MAE.
    """
    n_el = x_ref.size
    mae = np.linalg.norm(x_meas-x_ref)/n_el
    return mae

def nmse_freq(x_meas, x_ref):
    """ Computes the normalized mean squared error (NMSE) as a function of frequency.

    The input quantities are assumed to be matrices with :math:`N \\times N_{\\text{freq}}`
    entries. For details about the NMSE see :func:`nmse`.
    
    Parameters
    ----------
        x_meas : ndarray, shape (N, Nf)
            A matrix containing :math:`N` spectra of a measured quantity representing 
            a given solution or a reconstruction of a quantity of interest. Each spectrum
            has :math:`N_{\\text{freq}}` spectral lines.
        x_ref : ndarray, shape (N, Nf)
            A matrix containing the reference value for the measured quantity.
    
    Returns
    -------
        nnse : ndarray, shape (Nf,)
            Estimated NMSE spectrum.
    """
    _, nfreq = x_meas.shape
    nmse_freq = np.zeros(nfreq)
    for jf in np.arange(nfreq):
        nmse_freq[jf] = nmse(x_meas[:,jf], x_ref[:,jf])
    return nmse_freq