import numpy as np
from pyregtools.regchoice import csvd, gram_matrix

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