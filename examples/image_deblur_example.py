import numpy as np
import matplotlib.pyplot as plt
from pyregtools.toyproblems import ImageDeblur
from pyregtools.regsolvers import csvd, tikhonov
from pyregtools.regchoice import l_curve, gcv_lambda


size = 160
problem = ImageDeblur(image_name="moon.png", shape=(size, size))
problem.gaussian_psf(sigma=2.0)
problem.blur_separable()
problem.add_noise(snr=60, seed=42)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(problem.b_true, cmap="gray", vmin=0, vmax=1)
ax[0].set_title("Original image")
im = ax[1].imshow(problem.psf, cmap="viridis")
ax[1].set_title("Gaussian PSF")
ax[2].imshow(problem.b_noisy, cmap="gray", vmin=0, vmax=1)
ax[2].set_title("Blurred and noisy image")
for a in ax:
    a.axis("off")
fig.colorbar(im, ax=ax[1])
plt.tight_layout()


#%% Inverse problem
ux, sx, vx = csvd(problem.A_x)
uy, sy, vy = csvd(problem.A_y)
u = np.kron(uy, ux)
v = np.kron(vy, vx)
s = np.kron(sy, sx)

b_noisy = problem.b_noisy.ravel(order="C")
#lam_opt = l_curve(u, s, b_noisy, plotit = True, r_min=0.05)
lam_opt = gcv_lambda(u, s, b_noisy, plot_gcvfun = True, r_min=0.000005)
x_est = tikhonov(u, s, v, b_noisy, lam_opt)
x_est = np.reshape(x_est, problem.shape, order="C")

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(problem.b_true, cmap="gray", vmin=0, vmax=1)
ax[0].set_title("Original image")
ax[1].imshow(x_est, cmap="gray", vmin=0, vmax=1)
ax[1].set_title("Reconstructed image")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.show()