import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy as np
from . import cones as C
from scipy.sparse.linalg import aslinearoperator as aslo
import scipy
from block import block, block_diag

import cvxpy as cp
from time import time

def _unpack(data):
    return data['A'], data['b'], data['c']


def _proj_onto_C(x, cone, n):
    assert(x.ndim == 1)
    out = np.zeros(len(x))
    out[:n] = x[:n]
    out[n:-1] = _proj_onto_Kstar(x[n:-1], cone)
    out[-1] = np.maximum(0, x[-1])
    return out


_cones = ['f', 'l', 'q', 's', 'ep', 'ed', 'p']


def _proj_onto_Kstar(x, cone):
    assert(x.ndim == 1)
    out = np.zeros(len(x))
    i = 0
    for c in _cones:
        if _cone_exists(c, cone):
            c_len = _cone_len(c, cone)
            out[i:i + c_len] = _dual_cone_proj(x[i:i + c_len], c, cone)
            i += c_len

    assert i == len(x)
    return out


def _cone_exists(c, cone):
    if c not in cone:
        return False

    if isinstance(cone[c], int):
        return cone[c] > 0

    if isinstance(cone[c], list):
        return len(cone[c]) > 0

    raise ValueError(
        "Cannot interpret cone data for cone {} with data {}".format(c, cone[c]))


def _cone_len(c, cone):
    assert c in cone
    if isinstance(cone[c], int):
        if c in ('ep', 'ed'):
            return 3 * cone[c]
        else:
            return cone[c]

    if isinstance(cone[c], list):
        if c == 's':
            return sum(c0 * (c0 + 1) // 2 for c0 in cone[c])
        else:
            return sum(cone[c])

    raise ValueError(
        "Cannot calculate cone length for cone {} with data {}".format(c, cone[c]))


def _dual_cone_proj(x, c, cone):
    """ A projection onto the dual of the cone c """
    if c == 'f':
        # primal zero cone, dual free cone
        return C.P_f(x)
    if c == 'l':
        # linear cone (inequality constraints), self dual
        return C.P_l(x)
    if c == 'q':
        # second order cone (t,x), self dual
        return C.P_q(x, cone[c])
    if c == 's':
        # psd cone, self dual
        return C.P_s(x, cone[c])
    if c == 'ep':
        # primal exponential cone, dual exponential cone
        return C.P_ed(x, cone[c])
    if c == 'ed':
        # dual exponential cone, dual primal cone
        return C.P_ep(x, cone[c])
    if c == 'p':
        # power cone, dual power cone
        return C.P_p(x, cone[c])

    raise NotImplementedError


def _J_onto_Kstar(x, cone):
    assert(x.ndim == 1)
    J_cones = []
    i = 0
    existing_cones = [c for c in _cones if _cone_exists(c, cone)]
    cone_lengths = []
    for c in existing_cones:
        if _cone_exists(c, cone):
            c_len = _cone_len(c, cone)
            J_cones.append(_dual_cone_J(x[i:i + c_len], c, cone))
            i += c_len
            cone_lengths.append(c_len)
    assert i == len(x)
    return J_cones, cone_lengths


def _dual_cone_J(x, c, cone):
    """ Given a point x, return the Jacobian of the cone projection onto c """
    if c == 'f':
        return C.J_f(x)
    if c == 'l':
        return C.J_l(x)
    if c == 'q':
        return C.J_q(x, cone[c])
    if c == 's':
        return C.J_s(x, cone[c])
    # reminder: these are intentionally switched, since this is onto the dual
    # cone of c, so projection onto dual of ep is ed
    if c == 'ep':
        return C.J_ed(x, cone[c])
    if c == 'ed':
        return C.J_ep(x, cone[c])
    if c == 'p':
        return C.J_p(x, cone[c])

    raise NotImplementedError


def _compute_r(utilde, u, v, cone, n, IplusQ):
    """ Given current iterates, compute the resisduals """
    r_utilde = np.asarray(IplusQ.dot(utilde)).squeeze() - (u + v)
    r_u = u - _proj_onto_C(utilde - v, cone, n)
    r_v = utilde - u
    return r_utilde, r_u, r_v


def _extract_solution(u, v, n, zerotol):
    """ Extract the original cone problem solutions from the given ADMM iterates """
    if np.abs(u[-1]) < zerotol:
        scale = zerotol  # 1.0
    else:
        scale = u[-1]
    x = u[0:n] / scale
    s = v[n:-1] / scale
    y = u[n:-1] / scale
    return (x, s, y)

def _setup_benchmark():
    b = {
        'time' : [], 
        'fval' : [], 
        'error' : []
    }
    return b

def error_from_x(x, benchmark, m):
    """ From x, return beta """
    # prob, beta, betastar = benchmark
    # scs_output = {
    #     "info" : {
    #         'status': 'Solved', 
    #         'statusVal': 1, 
    #         'resPri': 1, 
    #         'resInfeas': 1, 
    #         'solveTime': 1, 
    #         'relGap': 1, 
    #         'iter': 1, 
    #         'dobj': 1, 
    #         'pobj': 1, 
    #         'setupTime': 1, 
    #         'resUnbdd': 1, 
    #         'resDual': 1},
    #     "y" : np.zeros(m),
    #     "x" : x,
    #     "s" : np.zeros(m)
    # }
    # prob.unpack_results(cp.SCS, scs_output)
    beta_from_x, betastar = benchmark
    return np.linalg.norm(beta_from_x(x) - betastar)
    # if isinstance(beta, cp.Variable): 
    #     return np.linalg.norm(beta.value - betastar)
    # else: 
    #     return np.sum(np.linalg.norm(b1.value-b2) 
    #                   for b1, b2 in zip(beta, betastar))

def update_benchmark(b, u, v, c, t, zerotol, benchmark, m): 
    n = len(c)
    b['time'].append(t)
    x, _, _ = _extract_solution(u, v, n, zerotol)
    b['fval'].append(c.T.dot(x))
    b['error'].append(error_from_x(x, benchmark, m))

def M_matvec(v, solve_IplusQ, k):
    out = np.zeros(3 * k)
    out[:k] = solve_IplusQ(v[:k])
    out[k:] = v[k:]
    return out

def J_matvec(x, n, m, k, d, ridge, IplusQ, J_cones, cone_lengths): 
    out = np.zeros(3*k)
    out[:k] = (1+ridge)*IplusQ.dot(x[:k]) - x[k:-k] - x[-k:]
    out[k:k+n] = -x[:n] + (1+ridge)*x[k:k+n] + x[-k:-k+n]
    # cone block
    i = 0
    for K_op, K_len in zip(J_cones, cone_lengths): 
        out[k+n+i:k+n+i+K_len] = -K_op(x[n+i:n+i+K_len]) + (1+ridge)*x[k+n+i:k+n+i+K_len] + K_op(x[2*k+n+i:2*k+n+i+K_len])
        i += K_len

    out[k+n+m] = -d*x[n+m] + (1+ridge)*x[k+n+m] + d*x[2*k+n+m]
    out[2*k:] = x[:k] - x[k:-k]
    return out

def newton_admm(data, cone, maxiters=100, admm_maxiters=1,
                gmres_maxiters=lambda i: i // 10 + 1, zerotol=1e-14,
                res_tol=1e-10, ridge=0, alpha=0.001, beta=0.5,
                verbose=0, benchmark=None):
    A, b, c = _unpack(data)
    assert(A.ndim == 2 and b.ndim == 1 and c.ndim == 1)
    m, n = A.shape
    k = n + m + 1

    # preconstruct IplusQ and do an LU decomposition
    Q = sp.bmat([[       None,         A.T, c[:, None]],
                 [         -A,        None, b[:, None]],
                 [-c[None, :], -b[None, :],       None]])
    IplusQ = (sp.eye(Q.shape[0]) + Q).tocsc()
    if IplusQ.nnz/(Q.shape[0]**2) < 0.1: 
        IplusQ_LU = sla.splu(IplusQ)   
        solve_IplusQ = lambda v: IplusQ_LU.solve(v)
    else: 
        IplusQ = IplusQ.todense()
        IplusQ_LU = scipy.linalg.lu_factor(IplusQ)
        solve_IplusQ = lambda v: scipy.linalg.lu_solve(IplusQ_LU, v)


    # create preconditioner linear op
    M_lo = sla.LinearOperator((3 * k, 3 * k), matvec=lambda v: M_matvec(v, solve_IplusQ, k))

    # Initialize ADMM variables
    utilde, u, v = np.zeros(k), np.zeros(k), np.zeros(k)
    utilde[-1] = 1.0
    u[-1] = 1.0
    v[-1] = 1.0

    Ik = sp.eye(k)
    In = sp.eye(n)
    
    # save benchmarks in a dictionary   
    b = _setup_benchmark()
    extra_iters = 1

    for iters in range(admm_maxiters):
        if benchmark is not None: 
            start_time = time()
        
        # do admm step
        utilde = solve_IplusQ(u + v)
        u = _proj_onto_C(utilde - v, cone, n)
        v = v - utilde + u
        # print("#0", utilde, u)
        # assert False

        if benchmark is not None: 
            update_benchmark(b, u, v, c, time() - start_time, zerotol, benchmark, m)

        if verbose and np.mod(iters, verbose) == 0:
            # If still running ADMM iterations, compute residuals
            r_utilde, r_u, r_v = _compute_r(utilde, u, v, cone, n, IplusQ)
            _r_utilde, _r_u, _r_v = np.linalg.norm(
                r_utilde), np.linalg.norm(r_u), np.linalg.norm(r_v)
            x, _, _ = _extract_solution(u, v, n, zerotol)
            objval = c.T.dot(x)
            print("%d/%d: r_utilde = %E, r_u = %E, r_v = %E, obj val c^T x = %E." %
                  (iters, admm_maxiters, _r_utilde, _r_u, _r_v, objval))

            if all(r < res_tol for r in (_r_utilde, _r_u, _r_v)):
                print("Stopping early, residuals < {}".format(res_tol))
                break

    for iters in range(maxiters):
        if benchmark is not None: 
            start_time = time()

        # do newton step
        r_utilde, r_u, r_v = _compute_r(utilde, u, v, cone, n, IplusQ)

        # create linear operator for Jacobian
        # This code is more readable but slower, and is left here
        # just for the understanding of the reader. 
        # d = np.array(utilde[-1] - v[-1] >=
        #              0.0).astype(np.float64).reshape(1, 1)
        # D = block_diag(_J_onto_Kstar((utilde - v)[n:-1], cone)[0], 
        #                arrtype=sla.LinearOperator)
        # D_lo = block_diag([In, D, d], arrtype=sla.LinearOperator)

        # J_lo = block([[IplusQ * (1 + ridge), '-I', '-I'],
        #               [-D_lo, Ik * (1 + ridge), D_lo],
        #               ['I', '-I', -Ik * ridge]], arrtype=sla.LinearOperator)

        # Instead of building it block-wise, it is more efficient to just 
        # create the linear operator directly as follows. 

        d = np.array(utilde[-1] - v[-1] >=
                     0.0).astype(np.float64).reshape(1, 1)
        J_cones, cone_lengths = _J_onto_Kstar((utilde - v)[n:-1], cone)
        # print("#1", u)
        J_lo = sla.LinearOperator((3 * k, 3 * k), matvec=
            lambda x: J_matvec(x, n, m, k, d, ridge, IplusQ, J_cones, cone_lengths))
        # approximately solve then newton step
        dall, info = sla.gmres(J_lo,
                               np.concatenate([r_utilde, r_u, r_v]),
                               M=M_lo,
                               tol=1e-12,
                               maxiter=gmres_maxiters(iters) + extra_iters)
        # print("#2")
        # assert False
        dutilde, du, dv = dall[0:k], dall[k:2 * k], dall[2 * k:]

        # backtracking line search
        t = 1.0
        r_all_norm = np.linalg.norm(np.concatenate([r_utilde, r_u, r_v]))
        while True:
            utilde0 = utilde - t * dutilde
            u0 = u - t * du
            v0 = v - t * dv

            r_utilde0, r_u0, r_v0 = _compute_r(
                utilde0, u0, v0, cone, n, IplusQ)

            if not ((np.linalg.norm(np.concatenate([r_utilde0, r_u0, r_v0])) >
                     (1 - alpha * t) * r_all_norm) and
                    (t >= 1e-4)):
                break

            t *= beta

        if t < 1e-4: 
            extra_iters += 1

        # update iterates
        utilde, u, v = utilde0, u0, v0
        ridge *= 0.9

        if benchmark is not None: 
            update_benchmark(b, u, v, c, time() - start_time, zerotol, benchmark, m)

        if verbose and np.mod(iters, verbose) == 0:
            # If still running ADMM iterations, compute residuals
            _r_utilde, _r_u, _r_v = np.linalg.norm(
                r_utilde), np.linalg.norm(r_u), np.linalg.norm(r_v)
            x, _, _ = _extract_solution(u, v, n, zerotol)
            objval = c.T.dot(x)
            print("%d/%d: r_utilde = %E, r_u = %E, r_v = %E, obj val c^T x = %E. (%d)" %
                  (iters, maxiters, _r_utilde, _r_u, _r_v, objval, extra_iters))

            if all(r < res_tol for r in (_r_utilde, _r_u, _r_v)):
                print("Stopping early, residuals < {}".format(res_tol))
                break

    x, s, y = _extract_solution(u, v, n, zerotol)
    return {
        'x': x,
        's': s,
        'y': y,
        'info': {
            'fstar': c.dot(x),
            'benchmark' : b
        }
    }
