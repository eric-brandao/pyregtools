import numpy as np
import matplotlib.pyplot as plt
from pyregtools.toyproblems.gravity import Gravity1D
from pyregtools.regsolvers import csvd, tikhonov
from pyregtools.regchoice import l_curve

#%% Set up the Foward problem
problem = Gravity1D(lb = 0, ub = 1, d = 0.5) # instantiate
problem.sample_rho_uniform(L = 32) # sample source field
#problem.sample_rho_gauss_legendre(L = 32)
problem.sample_g(M = 64) # sample measurement of g
problem.sens_mtx() # build the sensing matrix
problem.multi_density_sin(amplitude = [1.0, 0.5], freq = [0.5, 1.0]) # create a source term
problem.noiseless_meas() # compute true measurement
problem.add_noise(snr = 30) # add some noise

#%% Set up the inverse problem
u, s, v = csvd(problem.A)
lam_opt = l_curve(u, s, problem.b_noisy, plotit = False)
x_est = tikhonov(u, s, v, problem.b_noisy, lam_opt)

#%% Plot
fig, axs = plt.subplots(1, 2, figsize = (9,3))
axs[0].plot(problem.x, problem.x_true, '--k')
axs[0].plot(problem.x, x_est, color = 'dodgerblue')
axs[0].grid(linestyle = '--')
axs[0].set_xlim((0, 1))
axs[0].set_ylim((0, 2.5))
axs[0].set_xlabel(r'$x$')
axs[0].set_ylabel(r'$\rho(x)$ [kg/m]')
axs[0].set_title(r'Source field (cond($\mathbf{{A}} = {}$)'.format(s[0]/s[-1]))
axs[1].plot(problem.coord, problem.b_true, '--k',
            label = r'true meas. ($d = {}$ m)'.format(problem.d))
axs[1].plot(problem.coord, problem.b_noisy, label = r'noisy meas.', 
            color = 'm', alpha = 0.7)
axs[1].grid(linestyle = '--')
axs[1].set_xlim((0, 1))
axs[1].set_ylim((0, 1.5*np.amax(problem.b_noisy)))
axs[1].legend(loc = 'upper right')
axs[1].grid(True)
axs[1].set_xlabel(r'$x´$')
axs[1].set_ylabel(r'$g(x´)$  [m/s$^2$]')
axs[1].set_title(r'Measured acceleration (no noise)')
plt.tight_layout()
plt.show()