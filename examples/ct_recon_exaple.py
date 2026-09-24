import numpy as np
import matplotlib.pyplot as plt
from pyregtools.toyproblems.ct_recon import CTReconstruction
from pyregtools.regsolvers import csvd, tikhonov, least_sq, tsvd
from pyregtools.iterativesolvers import art_solver, cgls
from pyregtools.regchoice import l_curve
from pyregtools.utils import plot_picard

#%%
problem = CTReconstruction(shape=(32, 32), size=1.0)
problem.sensing_mtx(angles=np.linspace(0, 180, 32, endpoint=False),  n_rays=48)
problem.create_phantom()
problem.noiseless_meas()
problem.add_noise(photon_count=1e5, seed=0)
problem.plot_true_phantom()
problem.plot_noisy_sinogram()

#%% Least sqrs
x_est = least_sq(problem.A, problem.b_noisy)
x_est = np.reshape(x_est, problem.shape)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(problem.x_true, origin="lower")
ax[0].set_title("True")
ax[1].imshow(x_est, origin="lower")
ax[1].set_title("Least squares")
plt.tight_layout()

#%%
u,s,v = csvd(problem.A)
lam_opt = l_curve(u, s, problem.b_noisy, plotit=True)
x_est = tikhonov(u, s, v, problem.b_noisy, lambd_value=lam_opt)
x_est = np.reshape(x_est, problem.shape)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(problem.x_true, origin="lower")
ax[0].set_title("True")
ax[1].imshow(x_est, origin="lower")
ax[1].set_title("Tikhonov")
plt.tight_layout()

#%%
x0 = np.copy(x_est.flatten())
x_it = cgls(problem.A, problem.b_noisy, x0 = None, max_it = 200)
x_it = x_it[0][:]
x_est = np.reshape(x_it[:,30].real, problem.shape)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(problem.x_true, origin="lower")
ax[0].set_title("True")
ax[1].imshow(x_est, origin="lower")
ax[1].set_title("IT solver")
plt.tight_layout()
plt.show()
