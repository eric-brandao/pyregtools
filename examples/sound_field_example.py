import numpy as np
import matplotlib.pyplot as plt
from pyregtools.toyproblems.sound_field_2D import SoundField2D
from pyregtools.regsolvers import csvd, tikhonov
from pyregtools.regchoice import l_curve

#%% Set up the Foward problem
problem = SoundField2D(c0 = 340, freq = 2000) # instantiate
problem.set_mic_array(x_len = 0.3, z_len = 0.3, n_x = 15, n_z = 15)
#problem.compute_pres(theta_deg = (45, -45), amps = (1, 0.7))
problem.smooth_sf(factor = -2)
problem.add_noise(snr = 30, seed = 0)
problem.plot_measured_field(coord = problem.coord, pres = problem.b_noisy)
#plt.show()
#%% Set up the inverse problem
problem.get_sens_mtx(nwaves = 180)
u, s, v = csvd(problem.A)
lam_opt = l_curve(u, s, problem.b_noisy, plotit = False)
x_est = tikhonov(u, s, v, problem.b_noisy, lam_opt)
problem.plot_solution_field(x_est)
#plt.show()

#%% Reconstruction
coord_recon, pres_recon = problem.reconstruct_pres(x_est, x_len = 0.5, z_len = 0.5, 
                                                   n_x = 50, n_z = 50)
problem.plot_measured_field(coord = coord_recon, pres = pres_recon)
#plt.show()

ref = SoundField2D(c0 = 340, freq = problem.freq) # instantiate
ref.set_mic_array(x_len = 0.5, z_len = 0.5, n_x = 50, n_z = 50)
#ref.compute_pres(theta_deg = (45, -45), amps = (1, 0.7))
ref.smooth_sf(factor = -2)
ref.plot_measured_field(coord = ref.coord, pres = ref.b_true)
plt.show()


print(problem.A.shape)
print(problem.b_true.shape)
print(problem.b_noisy.shape)
print(coord_recon.shape)
print(pres_recon.shape)


