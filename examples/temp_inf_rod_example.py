import numpy as np
import matplotlib.pyplot as plt
from pyregtools.toyproblems.temp_semi_inf_rod import TempSemiInfRod
from pyregtools.regsolvers import csvd, tikhonov
from pyregtools.regchoice import gcv_lambda, l_curve


#%% Set up the Foward problem
problem = TempSemiInfRod(xm=0.85, alpha = 1.0, T0 = 100, tmax = 5, fs = 20) # instantiate
problem.sens_mtx() # build the sensing matrix
#problem.temp_step(time_step=1.0, T_source=400.0)
#problem.temp_square_pulse(time_init=1, time_end=2, T_source=400) # create a source term
problem.temp_hann_pulse(time_init=1.0, time_end=3.0, T_source=400.0) # create a source term
#problem.temp_random(std_temp=20.0, seed=0)
problem.noiseless_meas() # compute true measurement
problem.add_noise(snr = 30) # add some noise
#%% Set up the inverse problem
u, s, v = csvd(problem.A[1:, :])
lam_opt = gcv_lambda(u, s, problem.b_noisy[1:]-problem.T0, plot_gcvfun = True, r_min=0.004)
#lam_opt = l_curve(u, s, problem.b_noisy[1:]-problem.T0, plotit = True)
x_est = tikhonov(u, s, v, problem.b_noisy[1:]-problem.T0, lam_opt)
x_est += problem.T0
#%% Plot
fig, axs = plt.subplots(1, 2, figsize = (9,3))
axs[0].plot(problem.time[1:], problem.x_true, '--k')
axs[0].plot(problem.time[1:], x_est, color = 'dodgerblue', alpha = 0.5)
axs[0].grid(linestyle = '--')
axs[0].set_xlim((0, problem.time[-1]))
axs[0].set_ylim((0, problem.x_true.max()+25))
axs[0].set_xlabel(r'time [s]')
axs[0].set_ylabel(r'$T(t)$  [K]')
axs[0].set_title(r'Source field (cond($\mathbf{{A}} = {}$)'.format(s[0]/s[-1]))
axs[1].plot(problem.time, problem.b_true, '--k', label = r'true meas.')
axs[1].plot(problem.time, problem.b_noisy, label = r'noisy meas.', 
            color = 'm', alpha = 0.7)
axs[1].grid(linestyle = '--')
axs[1].set_xlim((0, problem.time[-1]))
#axs[1].set_ylim((problem.b_true.min()-10, problem.b_true.max()+10))
axs[1].legend()
axs[1].grid(linestyle = '--')
axs[1].set_xlabel(r'time [s]')
axs[1].set_ylabel(r'$T(t)$  [K]')
axs[1].set_title(r'Measured temperature ($x_m = {}$ m)'.format(problem.xm))
plt.tight_layout()
#%% Reconstruction
