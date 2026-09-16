import numpy as np
import matplotlib.pyplot as plt
from pyregtools.toyproblems.gravity import Gravity1D
from pyregtools.regsolvers import csvd, tikhonov
from pyregtools.regchoice import l_curve

#%% Set up the Foward problem
problem = Gravity1D(lb = 0, ub = 1, d = 0.5) # instantiate
problem.sample_rho_mp(L = 16) # sample source field
problem.sample_rho_gauss_legendre(L = 16)
problem.sample_g(M = 64) # sample measurement of g
problem.sens_mtx() # build the sensing matrix
problem.multi_density_sin(amplitude = [1.0, 0.5], kx = [np.pi, 2*np.pi]) # create a source term
problem.noiseless_meas() # compute true measurement
problem.add_noise(snr = 30) # add some noise

problem.plot_problem()
plt.show()

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
axs[1].plot(problem.coord, problem.b_true, '--k', label = r'true meas.')
axs[1].plot(problem.coord, problem.b_noisy, label = r'noisy meas.', 
            color = 'm', alpha = 0.7)
axs[1].grid(linestyle = '--')
axs[1].set_xlim((0, 1))
axs[1].set_ylim((0, 1.5*np.amax(problem.b_noisy)))
axs[1].legend(loc = 'upper right')
axs[1].grid(linestyle = '--')
axs[1].set_xlabel(r'$x´$')
axs[1].set_ylabel(r'$g(x´)$  [m/s$^2$]')
axs[1].set_title(r'Measured acceleration ($d = {}$ m)'.format(problem.d))
plt.tight_layout()
plt.show()

#%% Reconstruction
### True value
problem.d = 0.25
problem.noiseless_meas()
### reconstruction
g_recon = problem.A @ x_est

plt.figure(figsize = (6,3))
plt.plot(problem.coord, problem.b_true, '--k', label = 'True')
plt.plot(problem.coord, g_recon, color = 'dodgerblue', label = 'Reconstructed')
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel(r'$x´$')
plt.ylabel(r'$g(x´)$  [m/s$^2$]')
plt.xlim((0, 1))
plt.ylim((0, 1.5*np.amax(problem.b_true)))
plt.title(r'meas. vs. recon. acceleration ($d = {}$ m)'.format(problem.d))
plt.tight_layout()
plt.show()