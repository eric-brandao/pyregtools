import numpy as np
from pyregtools.toyproblems import Gravity1D
from pyregtools.toyproblems import ImageDeblur

def test_gravity_forward_problem():
    """ Basic test on the Gravity toy problem
    """
    problem = Gravity1D()
    problem.sample_rho_mp(L=20)
    problem.sample_g(M=30)
    problem.density_piecewise()
    problem.noiseless_meas()
    assert problem.A.shape == (30, 20)
    assert problem.x_true.shape == (20,)
    assert problem.b_true.shape == (30,)
    np.testing.assert_allclose(problem.b_true, problem.A @ problem.x_true)

def test_blur_methods_consistency():
    """ Basic test on the Image deblurring toy problem
    """
    problem = ImageDeblur(shape=(20, 20), image_name=None)
    problem.gaussian_psf(sigma=1.0)
    problem.build_blur_matrices()
    b_fft = problem.blur_fft().copy()
    b_sep = problem.blur_separable().copy()
    b_kron = problem.blur_kron().copy()
    np.testing.assert_allclose(b_fft, b_sep, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b_fft, b_kron, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(problem.b_blur, b_kron)