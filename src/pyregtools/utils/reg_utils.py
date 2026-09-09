
import numpy as np
import matplotlib.pyplot as plt

def plot_colvecs(U, rows = 4, cols = 4, figsize = (8,5), ylim = (-0.2,0.2)):
    """ Plot the column vectors of left or right singular vector matrices.

    Parameters
    ----------
        U : numpy ndarray
            left or right singular vector matrix
        rows : int
            number of rows subfigures in the figure
        cols : int
            number of columns subfigures in the figure
        figsize : tuple
            figure size
        ylim : tuple
            min, max value of the y axis
    """
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
    
def plot_picard(U,s,b, noise_norm = None, figsize = (5,4)):
    """ Make the Picard plot

    Parameters
    ----------
        U : numpy ndarray
            left singular vector matrix
        s : numpy 1darray
            singular values
        b : numpy 1darray
            your measurement vector (size: Nm x 1)
        noise_norm : None or float
            estimated noise norm (if you have one)
        figsize : tuple
            figure size
    """
    # condition number
    cond_number = s[0]/s[-1]    
    # beta
    beta = np.abs(U.T @ b)    
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
    
def nmse(x_sol, x_truth):
    """ computes the NMSE (normalized mean squared error)

    Parameters
    ----------
        x_sol : numpy 1darray
            solution
        x_sol : numpy 1darray
            ground truth
    Returns
    -------
        nnse : float
            estimated NMSE
    """
    nmse = (np.linalg.norm(x_sol-x_truth)/np.linalg.norm(x_truth))**2
    return nmse

def mae(x_sol, x_truth):
    """ computes the MAE (nmean absolute error)

    Parameters
    ----------
        x_sol : numpy 1darray
            solution
        x_sol : numpy 1darray
            ground truth
    Returns
    -------
        mae : float
            estimated MAE
    """
    n_el = x_truth.size
    mae = np.linalg.norm(x_sol-x_truth)/n_el
    return mae

def nmse_freq(x_sol, x_truth):
    """ computes the NMSE vs freq (normalized mean squared error)

    Parameters
    ----------
        x_sol : numpy ndarray
            solution arraged in Nvals x Nfreq 
        x_sol : numpy ndarray
            ground truth arraged in Nvals x Nfreq 
    Returns
    -------
        nnse : nympy 1dArray
            estimated NMSE vs freq
    """
    _, nfreq = x_sol.shape
    nmse_freq = np.zeros(nfreq)
    for jf in np.arange(nfreq):
        nmse_freq[jf] = nmse(x_sol[:,jf], x_truth[:,jf])
    return nmse_freq