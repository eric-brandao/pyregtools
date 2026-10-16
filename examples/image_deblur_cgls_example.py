import numpy as np
import matplotlib.pyplot as plt
import scipy
from pyregtools.toyproblems import ImageDeblur
from pyregtools.iterativesolvers import cgls
from pyregtools.regchoice import lcurve_discrete
from pyregtools.utils import nmse_fun

# Small image so we can also construct the explicit matrix.
problem = ImageDeblur(image_name="moon.png", shape=(256, 256))
problem.gaussian_psf(sigma=2.5)
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

num_iterations = 300
x_lop, sol_norms, res_norms = cgls(Aop, b_noisy, max_it=num_iterations) # LOP
kl = lcurve_discrete(res_norms, sol_norms, plotit=True)
#%%

error_hist = nmse_fun(x_lop, b_true[:, None], axis=0)   

ko = np.argmin(error_hist)
print(ko)
print(kl)

x_est = np.reshape(x_lop[:,kl], problem.shape, order="C")

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

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(problem.b_true, cmap="gray", vmin=0, vmax=1)
ax[0].set_title("Original image")
ax[1].imshow(x_est, cmap="gray", vmin=0, vmax=1)
ax[1].set_title("Reconstructed image (CGLS)")
for a in ax:
    a.axis("off")
plt.tight_layout()


print("Estimated optimal it. from error history: {} (NMSE = {:.6f})".format(ko, error_hist[ko]))
print("Estimated optimal it. from lcurve: {} (NMSE = {:.6f})".format(kl, error_hist[kl]))

plt.figure()
plt.semilogy(np.arange(len(error_hist)), error_hist, '-ok')
plt.axvline(ko, linestyle = '--', color = 'r')
plt.axvline(kl, linestyle = '--', color = 'b')
plt.title("Estimated optimal it. from error history: {} (NMSE = {:.6f})".format(ko, error_hist[ko]))
plt.grid(linestyle = '--')
plt.ylim((0.001, 0.01))
plt.show()

