import numpy as np
from . import PSD
import scipy as sp
import scipy.sparse.linalg as sla
from block import block, block_diag


def P_f(x):
    """ Projection onto the free cone (x in R) """
    return x


def J_f(x):
    n = len(x)
    return sla.LinearOperator((n, n), matvec=lambda v: v)


def P_l(x):
    """ Projection onto the linear cone (x >= 0) """
    return np.maximum(0.0, x)


def J_l(x):
    n = len(x)
    d = (x >= 0).astype(np.float64)
    return sla.LinearOperator((n, n), matvec=lambda v: d * v)


def P_q(x, c_lens):
    """ Projection of (t,x) onto the second order cone """
    i = 0
    out = np.zeros(len(x))
    for c_len in c_lens:
        x_ = x[i:i + c_len]

        x0 = x_[1:]
        x0norm = np.linalg.norm(x0)
        t0 = x_[0]

        if x0norm <= -t0:
            out[i:i + c_len] = 0
        elif x0norm <= t0:
            out[i:i + c_len] = x_
        else:
            out[i:i + c_len] = 0.5 * \
                (1.0 + t0 / x0norm) * np.hstack([x0norm, x0])

        i += c_len
    return out


def J_q(x, c_lens):
    i = 0
    l = []
    for c_len in c_lens:
        x_ = x[i:i + c_len]

        x0 = x_[1:]
        x0norm = np.linalg.norm(x0)
        t0 = x_[0]

        if x0norm <= -t0:
            def matvec(v): return np.zeros(c_len)
        elif x0norm <= t0:
            def matvec(v): return v
        else:
            matvec = _soc(x0, x0norm, t0)
        l.append(sla.LinearOperator((c_len, c_len), matvec=matvec))
        i += c_len
    assert i == len(x), "{}, {}".format(i, len(x))
    return block_diag(l, arrtype=sla.LinearOperator)


def _soc(x0, x0norm, t0):
    """ Because of late binding in python, we need to define the function f in 
    a different scope than the loop in J_q """
    def f(x):
        lr_scale = -t0 / (2.0 * x0norm**3)
        last_rowcol = 1.0 / (2.0 * x0norm) * x0
        my_diag = 0.5 + t0 / (2.0 * x0norm)

        out = np.zeros(len(x))
        vecint0 = x[0]
        vecinx0 = x[1:]
        out[1:] = lr_scale * (x0 * (x0.T.dot(vecinx0))) + \
            last_rowcol * vecint0 + my_diag * vecinx0
        out[0] = last_rowcol.T.dot(vecinx0) + 0.5 * vecint0
        return out
    return f


def P_ep(x, n_cones):
    """ Projection onto the primal exponential cone """
    assert len(x) // 3 == n_cones
    out = np.zeros(len(x))
    for i in range(0, len(x), 3):
        z = x[i:i + 3]

        if _in_Kexp(z):
            p = z
        elif _in_KexpD(z):
            p = np.zeros(3)
        elif (z[0] < 0) and (z[1] < 0):
            p = np.array([z[0], np.maximum(0, z[1]), np.maximum(0, z[2])])
        else:
            p = _solve_Kexp_proj(z)

        out[i:i + 3] = p
    return out


def _exp_Js(x, n_cones):
    """ Contrust list of Jacobian matrices for the exponential cone projections """
    assert len(x) // 3 == n_cones
    out = []
    for i in range(0, len(x), 3):
        z = x[i:i + 3]

        if _in_Kexp(z):
            J = np.eye(3)
        elif _in_KexpD(z):
            J = np.zeros((3, 3))
        elif (z[0] < 0) and (z[1] < 0):
            J = np.diag(np.array([1, z[1] >= 0, z[2] >= 0]))
        else:
            r, s, t = tuple(_solve_Kexp_proj(z))
            l = t - z[2]  # t - t0
            alpha = np.exp(r / s)
            beta = l * r / (s**2) * alpha

            J_inv = np.array([[alpha, (-r + s) / s * alpha, -1, 0],
                              [1 + l / s * alpha, -beta, 0, alpha],
                              [-beta, 1 + beta * r / s,
                                  0, (1 - r / s) * alpha],
                              [0, 0, 1, -1]])
            J = np.linalg.inv(J_inv)[0:3, 1:]

        out.append(J)
    return out


def J_ep(x, n_cones):
    Js = _exp_Js(x, n_cones)
    return block_diag(out, arrtype=sla.LinearOperator)


def P_ed(x, n_cones):
    """ Projection onto the dual exponential cone """
    return x + P_ep(-x, n_cones)


def J_ed(x, n_cones):
    Js = _exp_Js(-x, n_cones)
    I = np.eye(3)
    Js0 = [I - J for J in Js]
    return block_diag(Js0, arrtype=sla.LinearOperator)


def P_s(x, c_lens):
    """ Projection onto the positive semidefinite cone. 

    Since SCS assumes an off-diagonal scaling of sqrt(2), scale the diagonal
    terms by sqrt(2), perform the projection, and rescale the diagonal terms 
    back by 1/sqrt(2). This is what SCS does in their projection implementation. 
     """
    i = 0
    out = np.zeros(len(x))
    for n in c_lens:
        n0 = n * (n + 1) // 2
        l = [0]
        out[i:i + n0] = PSD.P(x[i:i + n0])
        i += n0
    assert i == len(x)
    return out


def J_s(x, c_lens):
    i = 0
    out = []
    for n in c_lens:
        n0 = n * (n + 1) // 2
        J = PSD.J(x[i:i + n0])
        out.append(sla.aslinearoperator(J))
        i += n0
    assert i == len(x)
    return block_diag(out, arrtype=sla.LinearOperator)


def P_p(x, p_params):
    """ Projection onto the dual of power cone """
    raise NotImplementedError


def J_p(x, n_cones):
    raise NotImplementedError


def _in_Kexp(z):
    """ Returns true if z is in the exponential cone """
    alpha, beta, delta = z
    if ((beta > 0) and (delta > 0)
            and (np.log(delta) >= np.log(beta) + alpha / beta)) \
            or ((alpha <= 0) and (np.abs(beta) < 1e-12) and (delta >= 0)):
        return True
    else:
        return False


def _in_KexpD(z):
    """ Returns true if z is in the dual of the exponential cone """
    alpha, beta, delta = -z
    if ((alpha < 0) and (delta > 0) and (np.log(-alpha) + beta / alpha <= 1 + np.log(delta))) \
            or ((np.abs(alpha) < 1e-12) and (beta >= 0) and (delta >= 0)):
        return True
    else:
        return False


def _solve_Kexp_proj(z):
    def get_g(ztilde_in): return ztilde_in[1] * \
        np.math.exp(ztilde_in[0] / ztilde_in[1]) - ztilde_in[2]

    def get_grad_cons_wrt_ztilde(ztilde_in): return np.array([np.math.exp(ztilde_in[0] / ztilde_in[1]),
                                                              np.math.exp(
                                                                  ztilde_in[0] / ztilde_in[1]) * (1 - ztilde_in[0] / ztilde_in[1]),
                                                              -1])

    def get_grad(ztilde_in, gamma_in): return -np.concatenate([ztilde_in - z + gamma_in * get_grad_cons_wrt_ztilde(ztilde_in),
                                                               [get_g(ztilde_in)]])

    def scaled_get_grad_cons_wrt_ztilde(ztilde_in, b): return np.array([np.math.exp(ztilde_in[0] / ztilde_in[1] - b),
                                                                        np.math.exp(
                                                                            ztilde_in[0] / ztilde_in[1] - b) * (1 - ztilde_in[0] / ztilde_in[1]),
                                                                        -np.math.exp(-b)])

    def scaled_get_g(ztilde_in, b): return ztilde_in[1] * np.math.exp(
        ztilde_in[0] / ztilde_in[1] - b) - ztilde_in[2] * np.math.exp(-b)

    def log_2_norm_grad(ztilde_in, gamma_in):
        r, s, t = tuple(ztilde_in)
        b = max(r / s, 0)
        inv_exp = np.exp(-b)
        sum_grad = np.sum(np.concatenate([ztilde_in * inv_exp - z * inv_exp + gamma_in * scaled_get_grad_cons_wrt_ztilde(ztilde_in, b),
                                          [scaled_get_g(ztilde_in, b)]])**2)
        if sum_grad <= 0:
            return np.float('-inf')
        return 2 * b + np.log(sum_grad)

    maxiters_newt_kexp = 100  # 10
    alpha = 0.001
    beta = 0.5

    ztilde = np.array([1.0, 1.0, np.math.exp(1)])

    gamma = 1.0
    for iters in range(maxiters_newt_kexp):
        # Compute Hessian (H) + pre-factor it (into LU, P), compute gradient
        # (g).
        hess_cons_wrt_ztilde = np.math.exp(ztilde[0] / ztilde[1]) * \
            np.array([[1.0 / ztilde[1], -ztilde[0] / ztilde[1]**2, 0],
                      [-ztilde[0] / ztilde[1]**2, ztilde[0]**2 / ztilde[1]**3, 0],
                      [0, 0, 0]])

        hess = np.bmat([[np.eye(3) + gamma * hess_cons_wrt_ztilde, get_grad_cons_wrt_ztilde(ztilde)[:, None]],
                        [get_grad_cons_wrt_ztilde(ztilde)[None, :], np.array([[0]])]])
        LU, P = sp.linalg.lu_factor(hess)

        grad = get_grad(ztilde, gamma)

        if get_g(ztilde) < 1e-12 and np.linalg.norm(grad) < 1e-12:
            break

        nt = sp.linalg.lu_solve((LU, P), grad)
        ztilde_nt = nt[0:-1]
        gamma_nt = nt[-1]

        # Backtracking line search.
        t = 1.0
        while ((ztilde[1] + t * ztilde_nt[1]) <= 0 or
               (log_2_norm_grad(ztilde + t * ztilde_nt, gamma + t * gamma_nt)
                > 2 * np.log(1 - alpha * t) + log_2_norm_grad(ztilde, gamma))):
            t = t * beta

        # Update.
        ztilde = ztilde + t * ztilde_nt
        gamma = gamma + t * gamma_nt

    return ztilde
