import numpy as np
import matplotlib
matplotlib.use("Agg")
from pyregtools.regsolvers import csvd, gram_matrix
from pyregtools.utils import nmse, mae, plot_colvecs, plot_picard
from pyregtools.regsolvers import tikhonov, tikhonov_analytic, least_sq


def test_svd_singular_values():
    """ Test csvd for a simple case
    """
    A = np.diag([4.0, 2.0, 1.0])
    U, s, V = csvd(A)
    assert np.allclose(s, [4.0, 2.0, 1.0])

def test_csvd_reconstruction_tall_matrix():
    """ Test csvd reconstruction for a simple tall complex matrix using csvd
    """
    rng = np.random.default_rng(42)
    A = rng.standard_normal((20, 10)) + 1j * rng.standard_normal((20, 10))
    U, s, V = csvd(A)
    A_rec = U @ np.diag(s) @ V.conj().T
    assert np.allclose(A_rec, A)

def test_csvd_reconstruction_wide_matrix():
    """ Test csvd reconstruction for a simple wide complex matrix using csvd
    """
    rng = np.random.default_rng(42)
    A = rng.standard_normal((10, 20)) + 1j * rng.standard_normal((10, 20))
    U, s, V = csvd(A)
    A_rec = U @ np.diag(s) @ V.conj().T
    assert np.allclose(A_rec, A)

def test_gram_matrix_complex():
    """ Test Gram matrix function
    """
    A = np.array([[1, 1j], [2, 1], [1j, 2]], dtype=complex)
    G_expected = np.array([[1, (2 - 1j) / 6], [(2 + 1j) / 6, 1]], dtype=complex)
    G, cohe = gram_matrix(A)
    cohe_expected = np.sqrt(5) / 6
    assert np.allclose(G, G_expected)
    assert np.isclose(cohe, cohe_expected)

def test_gram_mtx_zero_column():
    """ Test Gram matrix function when A has a zero column
    """
    A = np.array([[1, 0], [2, 0], [1j, 0]], dtype=complex)
    G_expected = np.array([[1, 0],[0, 0]], dtype=complex)
    G, cohe = gram_matrix(A)
    assert np.allclose(G, G_expected)

def test_error_metrics():
    """ Test normalized mean square error function
    """
    np.random.seed(0)
    x = np.random.normal(loc = 0, scale = 1, size = 100)
    nmse(x, x) == 0 and mae(x, x) == 0

def test_equiv_form():
    """ Test Tikhonov solvers when lambda = 0 against least-squares.
    """
    # create random problem
    np.random.seed(0)
    A = np.random.normal(loc = 0, scale = 1, size = (10,10))
    x = np.random.normal(loc = 0, scale = 1, size = 10)
    b = A @ x
    # least squares solution
    x_lsq = least_sq(A, b)
    # Analytic Tikhonov 
    x_tik_ana = tikhonov_analytic(A, b, 0)
    # SVD Tikhonov 
    U, s, V = csvd(A)
    x_tik_svd = tikhonov(U, s, V, b, 0)
    assert np.allclose(x_lsq, x_tik_ana, rtol=1e-6, atol=1e-8) and\
        np.allclose(x_lsq, x_tik_svd, rtol=1e-6, atol=1e-8)

def test_plots():
    """ test if the plots run without error.
    """
    A = np.random.normal(0, 1, size = (20, 20))
    b = np.random.normal(0, 1, size = A.shape[0])
    U,s,V = csvd(A) 
    plot_colvecs(U, rows = 4, cols = 4, figsize = (8,5), ylim = (-0.2,0.2))
    matplotlib.pyplot.close("all")
    plot_picard(U, s, b, noise_norm = 0.001)
    matplotlib.pyplot.close("all")
    plot_picard(U, s, b, noise_norm = None)
    matplotlib.pyplot.close("all")



