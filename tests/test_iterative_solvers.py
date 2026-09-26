import numpy as np
import scipy
from pyregtools.regsolvers import csvd
from pyregtools.utils import nmse, mae
import matplotlib.pyplot as plt
from pyregtools.iterativesolvers import cgls, landweber, landweber_cimmino
from pyregtools.toyproblems.sound_field_2D import SoundField2D
from pyregtools.toyproblems.image_deblur import ImageDeblur

def import_mat_data():
    """ import reference data from matlab and return relevant data
    """
    data = scipy.io.loadmat("tests/data/hansen_ref_data_gravity_is.mat")
    x = data["x"].squeeze()
    A = data["A"]
    b = data["b"].squeeze()
    b_noisy = data["b_noisy"].squeeze()
    x_true = data["x_true"].squeeze() 
    snr = data["snr"].squeeze() 
    n = data["n"].squeeze()
    X = data['X']
    res_norm = data["rho"].squeeze()
    sol_norm = data["eta"].squeeze()
    error_hist = data["error_hist"].squeeze()
    return x, A, b, b_noisy, x_true, snr, n, X, res_norm, sol_norm, error_hist

def test_cgls_true():
    """ test cgls solver against true value for noiseless measurement
    """
    x, A, b, b_noisy, x_true, snr, n, X, res_norm, sol_norm, error_hist = import_mat_data()
    X_py, sol_norms, res_norms = cgls(A, b, x0 = None, max_it = 20)
    nmse_v = np.zeros(X_py.shape[1])
    mae_v = np.zeros(X_py.shape[1])
    for k in range(X_py.shape[1]):
        nmse_v[k] = nmse(x_meas = X_py[:,k], x_ref = x_true)
        mae_v[k] = mae(x_meas = X_py[:,k], x_ref = x_true)
    assert np.all(nmse_v[8:] < 1e-4)
    assert np.all(mae_v[8:] < 1e-3)

def test_cgls_vs_matlab():
    """ test cgls solver against matlab implementation
    """
    x, A, b, b_noisy, x_true, snr, n, X, res_norm, sol_norm, error_hist = import_mat_data()
    X_py, sol_norms, res_norms = cgls(A, b_noisy, x0 = None, max_it = 4)
    assert np.allclose(X_py, X[:,:X_py.shape[1]], rtol=1e-6, atol=1e-8)

def test_cgls_cplx():
    """ test cgls solver for complex 2D sound field data
    """
    #setup toy problem
    problem = SoundField2D(c0 = 340, freq = 2000) 
    problem.set_mic_array(x_len = 0.3, z_len = 0.3, n_x = 15, n_z = 15)
    problem.smooth_sf(factor = -2)
    problem.add_noise(snr = 30, seed = 0)
    # Run CGLS solver - 8 iterations is optimal in this particular case
    problem.get_sens_mtx(nwaves = 180)
    X_py, sol_norms, res_norms = cgls(problem.A, problem.b_noisy, x0 = None, max_it = 8)
    # Reconstruct sound field with 8-th iteration
    _, p_recon = problem.reconstruct_pres(X_py[:,-1], x_len = 0.5, z_len = 0.5, n_x = 50, n_z = 50)
    # Create a reference sound field
    ref = SoundField2D(c0 = 340, freq = problem.freq) # instantiate
    ref.set_mic_array(x_len = 0.5, z_len = 0.5, n_x = 50, n_z = 50)
    ref.smooth_sf(factor = -2)
    # NMSE
    nmse_val = nmse(x_meas=p_recon, x_ref = ref.b_true)
    assert nmse_val < 5e-4

def test_cgls_linear_operator():
    """ Simple test of Linear Operator for CGLS
    """
    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    x_true = np.array([1.0, -1.0])
    b = A @ x_true
    Aop = scipy.sparse.linalg.LinearOperator(shape=A.shape,
        matvec=lambda x: A @ x, rmatvec=lambda y: A.conj().T @ y, dtype=A.dtype)
    x_matrix, _, _ = cgls(A, b, max_it=2)
    x_operator, _, _ = cgls(Aop, b, max_it=2)
    np.testing.assert_allclose(x_operator,x_matrix,rtol=1e-12,atol=1e-12)

def test_cgls_linear_operator_imgdeblurring():
    """ Image deblurring test of Linear Operator for CGLS
    """
    # Small image so we can also construct the explicit matrix.
    problem = ImageDeblur(image_name="moon.png", shape=(32, 32))
    problem.gaussian_psf(sigma=2.0)
    problem.blur_fft()
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

    # Construct the explicit matrix using the existing method.
    problem.blur_kron()
    A = problem.A

    # Basic numerical testing before CGLS
    rng = np.random.default_rng(42)
    x = rng.standard_normal(N)
    y = rng.standard_normal(N)

    # Forward-model equivalence
    np.testing.assert_allclose(Aop.matvec(x), A @ x, rtol=1e-12, atol=1e-12)
    # Adjoint equivalence
    np.testing.assert_allclose(Aop.rmatvec(y), A.T @ y, rtol=1e-12, atol=1e-12)
    # Adjoint identity
    np.testing.assert_allclose(np.vdot(Aop.matvec(x), y), np.vdot(x, Aop.rmatvec(y)),
        rtol=1e-12, atol=1e-12)

    # Test CGLS
    b = problem.b_blur.ravel()
    max_it = 12
    x_matrix, sol_matrix, res_matrix = cgls(A, b, max_it=max_it) # Full matrix
    x_operator, sol_operator, res_operator = cgls(Aop, b, max_it=max_it) # LOP
    np.testing.assert_allclose(x_operator, x_matrix, rtol=1e-8, atol=1e-10)
    #np.testing.assert_allclose(res_operator, res_matrix, rtol=1e-8, atol=1e-10)

def test_landweber_iterations():
    """ Test landweber solver with analytical iterations
    """
    A = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
    ])
    b = np.array([1.0, 2.0])
    x_sol, sol_norms, res_norms = landweber(A, b, omega=0.1, max_it=2)
    x1 = np.array([0.1, 0.4])
    x2 = x1 + 0.1 * np.array([0.9, 2.4])
    expected = np.column_stack((x1, x2))
    expected_sol_norms = np.linalg.norm(expected, axis=0)
    expected_res_norms = np.linalg.norm(A @ expected - b[:, None], axis=0)
    np.testing.assert_allclose(x_sol, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(sol_norms, expected_sol_norms)
    np.testing.assert_allclose(res_norms, expected_res_norms)
    assert np.isrealobj(x_sol)

def test_landweber_cplx():
    """ test Landweber solver for complex 2D sound field data
    """
    #setup toy problem
    problem = SoundField2D(c0 = 340, freq = 2000) 
    problem.set_mic_array(x_len = 0.3, z_len = 0.3, n_x = 15, n_z = 15)
    problem.smooth_sf(factor = -2)
    problem.add_noise(snr = 30, seed = 0)
    # Run CGLS solver - 8 iterations is optimal in this particular case
    problem.get_sens_mtx(nwaves = 180)
    A_norm = np.linalg.norm(problem.A, ord=2)
    omega = 1.0 / A_norm**2
    X_py, sol_norms, res_norms = landweber(problem.A, problem.b_noisy, omega = omega, 
                                           x0 = None, max_it = 20)
    # Reconstruct sound field with 8-th iteration
    coord_recon, p_recon = problem.reconstruct_pres(X_py[:,-1], x_len = 0.5, z_len = 0.5, n_x = 50, n_z = 50)
    # Create a reference sound field
    ref = SoundField2D(c0 = 340, freq = problem.freq) # instantiate
    ref.set_mic_array(x_len = 0.5, z_len = 0.5, n_x = 50, n_z = 50)
    ref.smooth_sf(factor = -2)
    # NMSE
    nmse_val = nmse(x_meas=p_recon, x_ref = ref.b_true)
    assert nmse_val < 5e-4

def test_landweber_linear_operator():
    """ Simple test of Linear Operator for Landweber
    """
    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    _,s,_ = csvd(A)
    omega = 1/s[0]**2
    x_true = np.array([1.0, -1.0])
    b = A @ x_true
    Aop = scipy.sparse.linalg.LinearOperator(shape=A.shape,
        matvec=lambda x: A @ x, rmatvec=lambda y: A.conj().T @ y, dtype=A.dtype)
    x_matrix, _, _ = landweber(A, b, omega = omega, x0 = None, max_it = 3)
    x_operator, _, _ = landweber(Aop, b, omega = omega, x0 = None, max_it = 3)
    np.testing.assert_allclose(x_operator,x_matrix,rtol=1e-12,atol=1e-12)

def test_landweber_cimmino_iterations():
    """ Test landweber-cimmino solver with analytical iterations
    """
    A = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
    ])
    b = np.array([1.0, 2.0])
    x_sol, sol_norms, res_norms = landweber_cimmino(A, b, D=None, omega=0.1, max_it=2)
    x1 = np.array([0.05, 0.05])
    x2 = x1 + 0.1 * np.array([0.475, 0.475])
    expected = np.column_stack((x1, x2))
    expected_sol_norms = np.linalg.norm(expected, axis=0)
    expected_res_norms = np.linalg.norm(A @ expected - b[:, None], axis=0)
    np.testing.assert_allclose(x_sol, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(sol_norms, expected_sol_norms)
    np.testing.assert_allclose(res_norms, expected_res_norms)
    assert np.isrealobj(x_sol)

def test_landweber_cimmino_cplx():
    """ test Landweber-Cimmino solver for complex 2D sound field data
    """
    #setup toy problem
    problem = SoundField2D(c0 = 340, freq = 2000) 
    problem.set_mic_array(x_len = 0.3, z_len = 0.3, n_x = 15, n_z = 15)
    problem.smooth_sf(factor = -2)
    problem.add_noise(snr = 30, seed = 0)
    # Run CGLS solver - 8 iterations is optimal in this particular case
    problem.get_sens_mtx(nwaves = 180)
    row_norms = np.linalg.norm(problem.A, axis=1)
    D = np.zeros(problem.A.shape[0])
    nonzero = row_norms > 0
    D[nonzero] = 1.0 / (problem.A.shape[0] * row_norms[nonzero]**2)
    B = np.sqrt(D)[:, None] * problem.A
    B_norm = np.linalg.norm(B, ord=2)
    omega = 1.0 / B_norm**2
    X_py, sol_norms, res_norms = landweber_cimmino(problem.A, problem.b_noisy, 
                                                   D=None, omega = omega, 
                                                   x0 = None, max_it = 20)
    # Reconstruct sound field with 8-th iteration
    coord_recon, p_recon = problem.reconstruct_pres(X_py[:,-1], x_len = 0.5, z_len = 0.5, n_x = 50, n_z = 50)
    # Create a reference sound field
    ref = SoundField2D(c0 = 340, freq = problem.freq) # instantiate
    ref.set_mic_array(x_len = 0.5, z_len = 0.5, n_x = 50, n_z = 50)
    ref.smooth_sf(factor = -2)
    # NMSE
    nmse_val = nmse(x_meas=p_recon, x_ref = ref.b_true)
    assert nmse_val < 5e-4
