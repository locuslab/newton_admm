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


def robust_pca_alnur(N, suppfrac, lam=1.0): 
    """ Create a robust PCA problem with a low rank matrix """

    # First, create a rank = "rk" matrix:
    # N = 10 # 500 # 300                     # Num examples (i.e., rows in X).
    p = N # 1000 # 600                    # Num features (i.e., cols of X).
    suppfrac = 0.1 # 0.1 # 0.01 # 0.05                  # Fraction of "p" that should be nonzero.

    K = ["nno", "nno", "nno", "nno", "nno", "psd"]       # Cones that each block of u_y belong to.  "nno" == nonnegative orthant, "soc" == second order cone, "exp" == exp cone, psd" == PSD cone.
    Klens = [N*p, N*p, 1, N*p, N*p, (N+p)**2]             # Length of each block of u_y.
    Kpsd_sizes = [[N, N, p, p]]                      # For each Kpsd constraint, we require a list of (in the following order):
                                           # num. rows in the upper left block
                                           # num. cols in the upper left block
                                           # num. cols in the upper right block
                                           # num. rows in the lower right block.
                                           # Create data.

    # First, create a rank = "rk" matrix:
    assert N == p                                  # This approach to creating a low-rank matrix requires N == p
    rk = int(round(N*0.5))
    assert rk <= min(N,p)
    Lstar = np.zeros((N,p))
    for i in range(rk):
        vi = np.random.randn(N,1)
        mati = vi.dot(vi.T)
        Lstar += mati
        
    # Then, create a sparse matrix:
    Mstar = np.random.randn(N,p)
    Mstar_vec = Mstar.T.ravel()

    nnz = int(np.floor((1.0-suppfrac)*N*p))        # Num. nonzeros
    assert nnz <= N*p
    idxes = np.random.randint(0, N*p, nnz)
    Mstar_vec[idxes] = 0
    # print "Number of zeros in solution is %d/%d entries." % (len(idxes), N*p)

    Mstar = np.reshape(Mstar_vec, (N,p)).T

    # Finally, sum the two matrices "L" and "M":
    X = Lstar + Mstar

    # Solve by cvxpy.
    vec = lambda matin: matin.T.ravel()

    zNp1 = np.zeros((N*p,1))
    ZNpNsq = np.zeros((N*p,N**2))
    ZNppsq = np.zeros((N*p,p**2))
    ZNpNp = np.zeros((N*p,N*p))
    z1Nsq = np.zeros((1,N**2))
    z1psq = np.zeros((1,p**2))
    z1Np = zNp1.T
    zNpluspsq1 = np.zeros(((N+p)**2,1))
    ZNpluspsqNp = np.zeros(((N+p)**2,N*p))

    o1Np = np.ones((1,N*p))

    IN = np.eye(N)
    Ip = np.eye(p)
    INp = np.eye(N*p)

    c = np.vstack([0.5*vec(IN)[:,None], 0.5*vec(Ip)[:,None], zNp1, zNp1, zNp1]).ravel()

    def create_zeros_mat_except_ij_one(nr, nc, i, j):
        mat = np.zeros((nr,nc))
        mat[i,j] = 1
        return mat

    row_offset = 0
    col_offset = 0
    G_W1_vecs = [vec(create_zeros_mat_except_ij_one(N+p, N+p, row_offset + i, col_offset + j))[:,None] 
                 for j in range(N) for i in range(N)]

    row_offset = N
    col_offset = N
    G_W2_vecs = [vec(create_zeros_mat_except_ij_one(N+p, N+p, row_offset + i, col_offset + j))[:,None] 
                 for j in range(p) for i in range(p)]


    row_offset = 0
    col_offset = N
    G_L_vecs = [vec(create_zeros_mat_except_ij_one(N+p, N+p, row_offset + i, col_offset + j) \
                    + create_zeros_mat_except_ij_one(N+p, N+p, row_offset + i, col_offset + j).T)[:,None] 
                for j in range(p) for i in range(N)]

    bot = np.hstack([np.hstack(G_W1_vecs), np.hstack(G_W2_vecs), ZNpluspsqNp, np.hstack(G_L_vecs), ZNpluspsqNp])

    # get it into SCS problem form
    mask = np.zeros((N+p,N+p), dtype=np.bool)
    ind = np.tril_indices(N+p)
    mask[ind] = True
    mask = vec(mask)
    bot = bot[mask]
    zNpluspsq1 = zNpluspsq1[mask]

    A = np.bmat([[ZNpNsq, ZNppsq, -INp, ZNpNp, -INp],
                 [ZNpNsq, ZNppsq, -INp, ZNpNp, INp],
                 [z1Nsq, z1psq, o1Np, z1Np, z1Np],
                 [ZNpNsq, ZNppsq, ZNpNp, INp, INp],
                 [ZNpNsq, ZNppsq, ZNpNp, -INp, -INp],
                 [-bot]])
       
    b = np.vstack([zNp1, zNp1, lam, vec(X)[:,None], -vec(X)[:,None], zNpluspsq1]).ravel()

    m = A.shape[0]
    # assert m == N*p + N*p + 1 + N*p + N*p + (N+p)**2
    # assert m == b.shape[0]

    n = A.shape[1]
    # assert n == N**2 + p**2 + N*p + N*p + N*p

    cumsums = np.cumsum([0] + Klens)
    Kidxes = [range(cumsums[i], cumsums[i+1]) for i in range(len(cumsums)-1)]

    Lstar_vec = vec(Lstar)
    Mstar_vec = vec(Mstar)

    print(A.shape, 4*N*p+1, N+p)

    return (None, None, {
        'A' : A, 
        'b' : b,
        'c' : c, 
        'dims' : {
            'l' : 4*N*p+1,
            's' : [N+p]
        },
        'beta_from_x' : lambda x: x[:N*N+p*p]
    })


def robust_pca(p, suppfrac, lam=1.0):
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

    Lhat = cp.Variable(p, p)
    Mhat = cp.Variable(p, p)
    prob = cp.Problem(cp.Minimize(cp.norm(Lhat, "nuc") + cp.sum_squares(Lhat)
        + cp.sum_squares(Mhat)),
                      [cp.norm(Mhat, 1) <= lam, Lhat + Mhat == X])

    data = prob.get_problem_data(cp.SCS)
    data['beta_from_x'] = cvxpy_beta_from_x(prob, (Lhat, Mhat), data['A'].shape[0])
    return ((Lhat, Mhat), prob, data)