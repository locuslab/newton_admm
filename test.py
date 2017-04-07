from newton_admm import newton_admm, problems, PSD

import numpy as np
import cvxpy as cp

np.random.seed(0)

def _s(o1, o2):
    return "Final objective values don't match, got {} but expected {}.".format(o1, o2)


def test_least_squares():
    """ This test will construct second order cone constraints """
    prob = problems.least_squares(10, 5)
    data = prob.get_problem_data(cp.SCS)
    out = newton_admm(data, data['dims'])
    cvx_out = prob.solve()

    assert np.allclose(out['info']['fstar'], cvx_out), _s(
        out['info']['fstar'], cvx_out)


def test_lp():
    """ This test will construct equality, inequality, and second order cone constraints """
    prob = problems.lp(30, 60)
    data = prob.get_problem_data(cp.SCS)
    out = newton_admm(data, data['dims'])
    cvx_out = prob.solve()

    assert np.allclose(out['info']['fstar'], cvx_out), _s(
        out['info']['fstar'], cvx_out)


def test_portfolio_opt():
    """ This test will construct equality, inequality, and second order cone constraints """
    prob = problems.portfolio_opt(10)
    data = prob.get_problem_data(cp.SCS)
    out = newton_admm(data, data['dims'])
    cvx_out = prob.solve()

    assert np.allclose(out['info']['fstar'], cvx_out), _s(
        out['info']['fstar'], cvx_out)


def test_logistic_regression():
    """ This test will construct inequality, and exponential cone constraints """
    prob = problems.logistic_regression(5, 2, 1.0)
    data = prob.get_problem_data(cp.SCS)
    out = newton_admm(data, data['dims'])
    cvx_out = prob.solve()

    assert np.allclose(out['info']['fstar'], cvx_out), _s(
        out['info']['fstar'], cvx_out)


def test_PSD():
    from numdifftools import Jacobian
    x = np.random.randn(6)
    J = Jacobian(lambda x: PSD.P(x))

    assert np.allclose(J(x), PSD.J(x))


def test_robust_pca():
    """ This test will construct positive semi-definite cone constraints """
    prob = problems.robust_pca(5, 0.5)
    data = prob.get_problem_data(cp.SCS)
    # solve the problem with ADMM instead for better accuracy
    out0 = newton_admm(data, data['dims'], admm_maxiters=4000, maxiters=4000)
    out = newton_admm(data, data['dims'])
    fstar, fstar0 = out['info']['fstar'], out0['info']['fstar']
    assert np.allclose(fstar, fstar0), _s(fstar, fstar0)


if __name__ == '__main__':
    test_least_squares()
    test_lp()
    test_portfolio_opt()
    test_logistic_regression()
    test_PSD()
    test_robust_pca()
