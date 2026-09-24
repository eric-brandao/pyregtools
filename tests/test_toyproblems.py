import numpy as np
from pyregtools.toyproblems import Gravity1D
from pyregtools.toyproblems import TempSemiInfRod
from pyregtools.toyproblems import MultiDecay
from pyregtools.toyproblems import SoundField2D
from pyregtools.toyproblems import ImageDeblur
from pyregtools.toyproblems import CTReconstruction

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

def test_temp_forward_problem():
    """ Basic test on the Temperature Inf Rod toy problem
    """
    problem = TempSemiInfRod(xm=1.0, alpha = 1.0, T0 = 100, tmax = 5, fs = 20)
    problem.sens_mtx() # build the sensing matrix
    problem.temp_step(time_step=1.0, T_source=400.0)
    b_analytical = problem.temp_analytical_step(time_step=1.0, T_source=400.0)
    problem.noiseless_meas() # compute true measurement
    problem.add_noise(snr = 30)
    assert problem.A.shape == (100, 99)
    assert problem.x_true.shape == (99,)
    assert problem.b_true.shape == (100,)
    assert problem.b_noisy.shape == (100,)
    np.testing.assert_allclose(problem.b_true, b_analytical)

def test_mdec_forward_problem():
    """ Basic test on the Multidecay toy problem
    """
    problem = MultiDecay(t_max = 2, fs = 100) # instantiate
    problem.compose_decay(t_decays = (0.5, 1.00), amps = (1.0, 0.25), amp_int_noise = 0.0001)
    problem.add_noise(snr = 30) # add some noise
    problem.sensing_mtx(time_decay_range = (1e-3, 3), num_cand_decay = 250,
                        include_int_noise = True)
    assert problem.A.shape == (200, 251)
    assert problem.b_true.shape == (200,)
    assert problem.b_noisy.shape == (200,)

def test_sf2d_forward_problem():
    """ Basic test on the Sound Field 2D toy problem
    """
    problem = SoundField2D(c0 = 340, freq = 2000) # instantiate
    problem.set_mic_array(x_len = 0.3, z_len = 0.3, n_x = 15, n_z = 15)
    problem.compute_pres(theta_deg = (45, -45), amps = (1, 0.7))
    problem.add_noise(snr = 30, seed = 0)
    problem.get_sens_mtx(nwaves = 180)
    coord_recon, pres_recon = problem.reconstruct_pres(np.ones(180), x_len = 0.5, z_len = 0.5, 
        n_x = 50, n_z = 50)
    assert problem.coord.shape == (225,2)
    assert problem.A.shape == (225, 180)
    assert problem.b_true.shape == (225,)
    assert problem.b_noisy.shape == (225,)
    assert coord_recon.shape == (2500, 2)
    assert pres_recon.shape == (2500,)

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

def test_diagonal_ray():
    """ Test1 CT recon
    """
    ct = CTReconstruction(shape=(4, 4), size=1.0)
    lengths = ct._ray_pixel_lengths(theta_deg=45.0, offset=0.0)
    lengths = lengths.reshape(ct.shape)
    expected = np.fliplr(np.eye(4)) * np.sqrt(2) / 4
    np.testing.assert_allclose(lengths, expected, atol=1e-14)

def test_offcenter_diagonal_ray():
    """ Test2 CT recon
    """
    ct = CTReconstruction(shape=(4, 4), size=1.0)
    ct.sensing_mtx(angles=[0, 45, 90], n_rays=5)
    offset = ct.offsets[3]
    expected_length = (np.sqrt(2) * ct.size- 2 * abs(offset))
    expected = (
        np.fliplr(np.eye(4))
        * np.sqrt(2) / 4
    )
    np.testing.assert_allclose(ct.A[7].reshape(ct.shape), expected, atol=1e-14)
    np.testing.assert_allclose(ct.A[8].sum(), expected_length, atol=1e-14)
    assert ct.A.shape == (15, 16)

def test_ray_theta_0():
    ct = CTReconstruction(shape=(4, 4), size=1.0)
    lengths = ct._ray_pixel_lengths(theta_deg=0.0, offset=0.125).reshape(ct.shape)
    expected = np.zeros((4, 4))
    expected[:, 2] = 0.25
    np.testing.assert_allclose(lengths, expected, atol=1e-14)


def test_ray_theta_90():
    ct = CTReconstruction(shape=(4, 4), size=1.0)
    lengths = ct._ray_pixel_lengths(theta_deg=90.0, offset=0.125).reshape(ct.shape)
    expected = np.zeros((4, 4))
    expected[2, :] = 0.25
    np.testing.assert_allclose(lengths, expected, atol=1e-14)

def test_ray_theta_45():
    ct = CTReconstruction(shape=(4, 4), size=1.0)
    lengths = ct._ray_pixel_lengths(theta_deg=45.0, offset=0.0).reshape(ct.shape)
    expected = (np.fliplr(np.eye(4)) * np.sqrt(2) / 4)
    np.testing.assert_allclose(lengths, expected, atol=1e-14)

def test_sensing_matrix_nonnegative():
    ct = CTReconstruction(shape=(8, 8), size=1.0)
    ct.sensing_mtx(angles=np.linspace(0, 180, 8, endpoint=False), n_rays=12)
    assert np.all(ct.A >= 0)

def test_forward_model():
    ct = CTReconstruction(shape=(16, 16), size=1.0)
    ct.create_phantom()
    ct.sensing_mtx(angles=np.linspace(0, 180, 16, endpoint=False), n_rays=24)
    ct.noiseless_meas()
    ct.add_noise(photon_count=1e5, seed=0)
    b1 = ct.b_noisy.copy()
    ct.add_noise(photon_count=1e5, seed=0)
    b2 = ct.b_noisy.copy()

    assert ct.x_true.shape == (16, 16)
    assert ct.sinogram_true.shape == (16, 24)
    assert ct.sinogram_noisy.shape == (16, 24)
    assert np.all(ct.x_true >= 0)
    assert np.any(ct.x_true > 0)
    assert ct.b_true.shape == (384,)
    assert ct.b_noisy.shape == (384,)
    assert np.all(ct.b_true >= 0)
    assert np.any(ct.b_true > 0)
    assert np.all(np.isfinite(ct.b_noisy))
    np.testing.assert_array_equal(b1, b2)