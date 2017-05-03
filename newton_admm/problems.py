import numpy as np
import cvxpy as cp
import scipy as sp


def least_squares(m, n):
    """ Create a least squares problem with m datapoints and n dimensions """
    A = np.random.randn(m, n)
    _x = np.random.randn(n)
    b = A.dot(_x)

    x = cp.Variable(n)
    return (x, cp.Problem(cp.Minimize(cp.sum_squares(A * x - b) + cp.norm(x, 2))))


def lp(m, n):
    """ Create a linear program with random equality constraints and 
    non-negative inequality constraints """
    _x = np.maximum(0.0, np.random.randn(n))

    # inequality constriants
    A1 = -np.eye(n)
    b1 = np.zeros(n)

    # equality constraints
    A2 = np.random.randn(m, n) 
    b2 = A2.dot(_x)

    # objective
    nu = np.random.randn(m)
    lam = np.random.rand(n)
    idxes = np.where(_x > 0)
    lam[idxes] = 0
    c = np.ravel(-A2.T.dot(nu) + lam)

    # create standard form cone parameters
    A = np.vstack([A1,A2,-A2])
    b = np.vstack([b1[:,None], b2[:,None], -b2[:,None]]).ravel()

    x = cp.Variable(n)
    return (x, cp.Problem(cp.Minimize(c.T * x),
                          [b1 - A1 * x >= 0, A2 * x == b2]), {
        'A' : A, 
        'b' : b,
        'c' : c, 
        'dims' : {
            'l' : 2*m+n
        },
        'beta_from_x' : lambda x: x
    })


def portfolio_opt(p):
    """ Create a portfolio optimization problem with p dimensions """
    temp = np.random.randn(p, p)
    Sigma = temp.T.dot(temp)


    Sigma_sqrt = sp.linalg.sqrtm(Sigma)
    o = np.ones((p, 1))

    # Create standard form cone problem
    Zp1 = np.zeros((p,1))
    
    # setup for cone problem
    c = np.vstack([Zp1, np.array([[1.0]])]).ravel()

    G1 = sp.linalg.block_diag(2.0*Sigma_sqrt, -1.0)
    q = np.vstack([Zp1, np.array([[1.0]])])
    G2 = np.hstack([o.T, np.array([[0.0]])])
    G3 = np.hstack([-o.T, np.array([[0.0]])])

    h = np.vstack([Zp1, np.array([[1.0]])])
    z = 1.0

    A = np.vstack([G2, G3, -q.T, -G1 ])
    b = np.vstack([1.0, -1.0, np.array([[z]]), h]).ravel()

    betahat = cp.Variable(p)
    return (betahat, cp.Problem(cp.Minimize(cp.quad_form(betahat, Sigma)),
                                [o.T * betahat == 1]), {
        'A' : A, 
        'b' : b,
        'c' : c, 
        'dims' : {
            'l' : 2,
            'q' : [p+2]
        },
        'beta_from_x' : lambda x: x[:p]
    })


def cvxpy_beta_from_x(prob, beta, m): 
    def beta_from_x(x):
        """ From x, return beta """
        scs_output = {
            "info" : {
                'status': 'Solved', 
                'statusVal': 1, 
                'resPri': 1, 
                'resInfeas': 1, 
                'solveTime': 1, 
                'relGap': 1, 
                'iter': 1, 
                'dobj': 1, 
                'pobj': 1, 
                'setupTime': 1, 
                'resUnbdd': 1, 
                'resDual': 1},
            "y" : np.zeros(m),
            "x" : x,
            "s" : np.zeros(m)
        }
        prob.unpack_results(cp.SCS, scs_output)
        if isinstance(beta, cp.Variable):
            return beta.value
        else:
            return tuple(b.value for b in beta)
    return beta_from_x

def logistic_regression(N, p, suppfrac):
    """ Create a logistic regression problem with N examples, p dimensions,
    and at most suppfrac of the optimal solution to be non-zero. """
    X = np.random.randn(N, p)

    betastar = np.random.randn(p)
    nnz = int(np.floor((1.0 - suppfrac) * p))        # Num. nonzeros
    assert nnz <= p
    idxes = np.random.randint(0, p, nnz)
    betastar[idxes] = 0

    probplus1 = 1.0 / (1.0 + np.exp(-X.dot(betastar)))
    y = np.random.binomial(1, probplus1)

    lam = 1.0  # 1.0

    # Solve by ECOS.
    betahat = cp.Variable(p)
    logloss = sum(cp.log_sum_exp(
        cp.hstack(0, y[i] * X[i, :] * betahat)) for i in range(N))
    prob = cp.Problem(cp.Minimize(logloss + lam * cp.norm(betahat, 1)))

    X = np.random.randn(N, p)

    betastar = np.random.randn(p)
    nnz = int(np.floor((1.0 - suppfrac) * p))        # Num. nonzeros
    assert nnz <= p
    idxes = np.random.randint(0, p, nnz)
    betastar[idxes] = 0

    probplus1 = 1.0 / (1.0 + np.exp(-X.dot(betastar)))
    y = np.random.binomial(1, probplus1)

    lam = 1.0  # 1.0

    # Solve by ECOS.
    betahat = cp.Variable(p)
    logloss = sum(cp.log_sum_exp(
        cp.hstack(0, y[i] * X[i, :] * betahat)) for i in range(N))
    prob = cp.Problem(cp.Minimize(logloss + lam * cp.norm(betahat, 1)))

    data = prob.get_problem_data(cp.SCS)
    data['beta_from_x'] = cvxpy_beta_from_x(prob, betahat, data['A'].shape[0])
    return (betahat, prob, data)


def robust_pca(p, suppfrac):
    """ Create a robust PCA problem with a low rank matrix """

    # First, create a rank = "rk" matrix:
    rk = int(round(p * 0.5))
    assert rk <= min(p, p)
    Lstar = np.zeros((p, p))
    for i in range(rk):
        vi = np.random.randn(p, 1)
        mati = vi.dot(vi.T)
        Lstar += mati

    # Then, create a sparse matrix:
    Mstar = np.random.randn(p, p)
    Mstar_vec = Mstar.T.ravel()

    nnz = int(np.floor((1.0 - suppfrac) * p * p))        # Num. nonzeros
    assert nnz <= p * p
    idxes = np.random.randint(0, p * p, nnz)
    Mstar_vec[idxes] = 0
    Mstar = np.reshape(Mstar_vec, (p, p)).T

    # Finally, sum the two matrices "L" and "M":
    X = Lstar + Mstar

    lam = 1.0

    Lhat = cp.Variable(p, p)
    Mhat = cp.Variable(p, p)
    prob = cp.Problem(cp.Minimize(cp.norm(Lhat, "nuc") + cp.sum_squares(Lhat)),
                      [cp.norm(Mhat, 1) <= lam, Lhat + Mhat == X])

    data = prob.get_problem_data(cp.SCS)
    data['beta_from_x'] = cvxpy_beta_from_x(prob, (Lhat, Mhat), data['A'].shape[0])
    return ((Lhat, Mhat), prob, data)