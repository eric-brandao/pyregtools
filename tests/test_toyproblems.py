import numpy as np
from pyregtools.toyproblems import Gravity1D

def test_gravity_forward_problem():
    problem = Gravity1D()

    problem.sample_rho_mp(L=20)
    problem.sample_g(M=30)
    problem.density_piecewise()
    problem.noiseless_meas()
    assert problem.A.shape == (30, 20)
    assert problem.x_true.shape == (20,)
    assert problem.b_true.shape == (30,)
    np.testing.assert_allclose(problem.b_true, problem.A @ problem.x_true)