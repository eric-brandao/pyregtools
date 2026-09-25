import numpy as np
import matplotlib.pyplot as plt
from pyregtools.toyproblems import ImageDeblur
from pyregtools.regsolvers import csvd, tikhonov
from pyregtools.regchoice import l_curve, gcv_lambda


size = 64
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
#plt.show()

#%%
import scipy
from pyregtools.iterativesolvers import cgls
# Small image so we can also construct the explicit matrix.
problem = ImageDeblur(image_name="moon.png", shape=(256, 256))
problem.gaussian_psf(sigma=2.0)
problem.blur_fft()
problem.add_noise(snr=60, seed=42)
b_noisy = problem.b_noisy.ravel(order="C")
b_true = problem.b_true.ravel(order="C")
M, L = problem.shape
N = M * L
H = problem.H_fft

def matvec(x):
    X = x.reshape(M, L)
    B = np.fft.ifft2(H * np.fft.fft2(X))
    return B.real.ravel()

def rmatvec(y):
    Y = y.reshape(M, L)
    X = np.fft.ifft2(np.conj(H) * np.fft.fft2(Y))
    return X.real.ravel()

Aop = scipy.sparse.linalg.LinearOperator(shape=(N, N), matvec=matvec, rmatvec=rmatvec,
    dtype=float)

num_iterations = 200
x_lop, _, _ = cgls(Aop, b_noisy, max_it=num_iterations) # LOP
x_est = np.reshape(x_lop[:,100], problem.shape, order="C")

error_hist = np.zeros(num_iterations)
for k in range(num_iterations):
    error_hist[k] = np.linalg.norm(x_lop[:,k]-b_true)/np.linalg.norm(b_true)

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

plt.figure()
plt.plot(error_hist)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(problem.b_true, cmap="gray", vmin=0, vmax=1)
ax[0].set_title("Original image")
ax[1].imshow(x_est, cmap="gray", vmin=0, vmax=1)
ax[1].set_title("Reconstructed image (CGLS)")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.show()