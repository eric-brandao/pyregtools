import numpy as np
import matplotlib.pyplot as plt
from pyregtools.toyproblems.multi_decay import MultiDecay
from pyregtools.regsolvers import csvd, tikhonov, cvx_reg
from pyregtools.regchoice import l_curve

#%% Set problem
problem = MultiDecay(t_max = 2, fs = 100) # instantiate
problem.compose_decay(t_decays = (0.5, 1.00), amps = (1.0, 0.25), amp_int_noise = 0.0001)
problem.add_noise(snr = 30) # add some noise
print("Last value of true edc: {}".format(problem.b_true[-1]))
problem.plot_edc()

#%% Inverse problem setting
problem.sensing_mtx(time_decay_range = (1e-3, 3), num_cand_decay = 250,
                    include_int_noise = True)
col_norms = np.linalg.norm(problem.A, axis=0)
B = problem.A @ np.diag(1/col_norms)

print(problem.A.shape)
print(problem.b_true.shape)
print(problem.b_noisy.shape)


plt.figure(figsize = (6,3))
#plt.stem(col_norms)
plt.stem(np.linalg.norm(B, axis=0))
plt.yscale('log')
plt.grid(linestyle = '--')

#%% Inverse problem estimation
x_est = cvx_reg(problem.A, problem.b_noisy, 0.1, is_lasso = True, is_complex = False)
#x_est = cvx_reg(B, problem.b_noisy, 2, is_lasso = True, is_complex = False)
#x_est = np.diag(1/col_norms) @ x_est

plt.figure(figsize = (6,4))
plt.plot(problem.t_decs, x_est[:len(problem.t_decs)])
plt.grid(linestyle = '--')
plt.xlabel("Decay Times [s]")
plt.ylabel("Amplitudes [-]")
plt.tight_layout()

#%% Reconstruction
edc_recon = problem.A @ x_est

plt.figure(figsize = (6,4))
plt.plot(problem.time, 10*np.log10(problem.b_true/np.amax(problem.b_true)), '--', 
            color = 'grey', linewidth = 1.0, label = "True EDC")
plt.plot(problem.time, 10*np.log10(problem.b_noisy/np.amax(problem.b_noisy)), ':', 
        color = 'k', linewidth = 1.0, label = "Noisy EDC")
plt.plot(problem.time, 10*np.log10(edc_recon/np.amax(edc_recon)), '-', 
        color = 'royalblue', linewidth = 1.2, label = "Reconstructed EDC")
plt.legend()
plt.grid(linestyle = '--')
plt.xlim([0, 2*problem.t_decays[1]])
plt.ylim([-60, 2])
plt.xlabel("Time [s]")
plt.ylabel("EDC [dB]")
plt.tight_layout()
plt.show()
