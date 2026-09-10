import numpy as np
from scipy.io import loadmat
from pyregtools.regchoice import csvd, l_curve_new, gcv_lambda, discrep, ncp
from pyregtools.regsolvers import tikhonov, tikhonov_analytic, sklearn_ridge, sklearn_ridge_c
from pyregtools.regsolvers import cvx_reg, tsvd, ssvd, cvx_constrained
from pyregtools.utils import nmse
import matplotlib.pyplot as plt

def import_mat_data():
    """ import reference data from matlab and return relevant data
    """
    data = loadmat("tests/data/hansen_ref_data_gravity.mat")
    x = data["x"].squeeze()
    A = data["A"]
    b = data["b"].squeeze()
    b_noisy = data["b_noisy"].squeeze()
    x_true = data["x_true"].squeeze() 
    snr = data["snr"].squeeze() 
    n = data["n"].squeeze()
    lam_dp = data["lam_dp"].squeeze()
    lam_lc = data["lam_lc"].squeeze()
    lam_gcv = data["lam_gcv"].squeeze()
    lam_ncp = data["lam_ncp"].squeeze()
    x_tik = data["x_tik"].squeeze() 
    x_k = data["x_k"].squeeze() 
    labda = data["labda"].squeeze()
    num_svd_comp = data["num_svd_comp"].squeeze()
    return x, A, b, b_noisy, x_true, snr, n, lam_dp, lam_lc, lam_gcv, lam_ncp, x_tik, x_k, labda, num_svd_comp 

def test_dp():
    """ test discrepancy principle against Hansen's implementation
    """
    x, A, b, b_noisy, x_true, snr, n,\
        lam_dp, lam_lc, lam_gcv, lam_ncp, x_tik, x_k, labda, num_svd_comp = import_mat_data()
    U, s, V = csvd(A)
    x_dp, lam_dp_py = discrep(U, s, V, b_noisy, 1.0*np.linalg.norm(n))
    assert np.isclose(lam_dp_py, lam_dp, rtol=1e-3)

def test_lc():
    """ test L-curve principle against Hansen's implementation
    """
    x, A, b, b_noisy, x_true, snr, n,\
        lam_dp, lam_lc, lam_gcv, lam_ncp, x_tik, x_k, labda, num_svd_comp  = import_mat_data()
    U, s, V = csvd(A)
    lam_lc_py = l_curve_new(U, s, b_noisy)
    assert np.isclose(lam_lc_py, lam_lc, rtol=1e-3)

def test_gcv():
    """ test GCV principle against Hansen's implementation
    """
    x, A, b, b_noisy, x_true, snr, n,\
        lam_dp, lam_lc, lam_gcv, lam_ncp, x_tik, x_k, labda, num_svd_comp  = import_mat_data()
    U, s, V = csvd(A)
    lam_gcv_py = gcv_lambda(U, s, b_noisy)
    assert np.isclose(lam_gcv_py, lam_gcv, rtol=1e-3)

def test_ncp():
    """ test NCP principle against Hansen's implementation
    """
    x, A, b, b_noisy, x_true, snr, n,\
        lam_dp, lam_lc, lam_gcv, lam_ncp, x_tik, x_k, labda, num_svd_comp  = import_mat_data()
    U, s, V = csvd(A)
    lam_ncp_py = ncp(U, s, b_noisy)
    assert np.isclose(lam_ncp_py, lam_ncp, rtol=1e-3)

def test_tikhonov():
    """ Test tikhonov function
    """
    x, A, b, b_noisy, x_true, snr, n, lam_dp, lam_lc, lam_gcv, lam_ncp,\
        x_tik, x_k, labda, num_svd_comp  = import_mat_data()
    U, s, V = csvd(A)
    x_tik_py = tikhonov(U, s, V, b_noisy, labda)
    nmse = np.linalg.norm(x_tik_py-x_tik)/np.linalg.norm(x_tik)
    assert np.allclose(x_tik_py, x_tik, rtol=1e-6, atol=1e-8)

def test_tikhonov_analytic():
    """ Test tikhonov analytic function
    """
    x, A, b, b_noisy, x_true, snr, n, lam_dp, lam_lc, lam_gcv, lam_ncp,\
        x_tik, x_k, labda, num_svd_comp  = import_mat_data()
    x_tik_py = tikhonov_analytic(A, b_noisy, labda)
    assert np.allclose(x_tik_py, x_tik, rtol=1e-6, atol=1e-8)

def test_sklearn_ridge():
    """ Test sklearn_ridge function
    """
    x, A, b, b_noisy, x_true, snr, n, lam_dp, lam_lc, lam_gcv, lam_ncp,\
        x_tik, x_k, labda, num_svd_comp  = import_mat_data()
    x_tik_py = sklearn_ridge(A, b_noisy, labda)
    assert nmse(x_tik_py, x_tik) < 0.05

def test_sklearn_ridge_c():
    """ Test sklearn_ridge_c function
    """
    x, A, b, b_noisy, x_true, snr, n, lam_dp, lam_lc, lam_gcv, lam_ncp,\
        x_tik, x_k, labda, num_svd_comp  = import_mat_data()
    x_tik_py = sklearn_ridge_c(A, b_noisy, labda)
    assert nmse(x_tik_py, x_tik) < 0.05

def test_tikhonov_cvx():
    """ Test tikhonov cvx function. 

    Test is done via nmse, since the solver is not as good as ref. tikhonov.
    """
    x, A, b, b_noisy, x_true, snr, n, lam_dp, lam_lc, lam_gcv, lam_ncp,\
        x_tik, x_k, labda, num_svd_comp  = import_mat_data()
    x_tik_py = cvx_reg(A, b, labda, is_lasso = False, is_complex = False)
    assert nmse(x_tik_py, x_tik) < 0.05

def test_tsvd():
    """ Test TSVD function
    """
    x, A, b, b_noisy, x_true, snr, n, lam_dp, lam_lc, lam_gcv, lam_ncp,\
        x_tik, x_k, labda, num_svd_comp  = import_mat_data()
    U, s, V = csvd(A)
    x_k_py = tsvd(U, s, V, b_noisy , num_svd_comp)
    assert np.allclose(x_k_py, x_k, rtol=1e-6, atol=1e-8)

def test_ssvd():
    """ Test SSVD function
    """
    x, A, b, b_noisy, x_true, snr, n, lam_dp, lam_lc, lam_gcv, lam_ncp,\
        x_tik, x_k, labda, num_svd_comp  = import_mat_data()
    U, s, V = csvd(A)
    x_s_py = ssvd(U, s, V, b_noisy , np.linalg.norm(n))
    assert np.allclose(x_s_py, x_k, rtol=1e-6, atol=1e-8)

