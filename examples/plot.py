
import numpy as np

import matplotlib
matplotlib.use('svg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt


def plotter(newton_out, scs_out, fstar, name, ours="Newton-ADMM", 
            xmax=None, ymin=1e-12, ymax=1e-2, lw=3, fontsize=14, 
            loc= "upper right"): 
    newton_out = newton_out['info']['benchmark']
    scs_out = scs_out['info']['benchmark']

    solve_times_SCS_ret = np.cumsum(scs_out['time'])
    solve_times_newt_ret = np.cumsum(newton_out['time'])
    errors_SCS_ret = scs_out['error']
    errors_newt_ret = newton_out['error']
    subopts_SCS = np.abs(fstar - np.array(scs_out['fval']))
    subopts_newt = np.abs(fstar - np.array(newton_out['fval']))

    # Make suboptimality plot.
    fig,ax = plt.subplots()
    ax.semilogy(solve_times_SCS_ret, subopts_SCS, lw=lw, color="blue", label="SCS")
    ax.semilogy(solve_times_newt_ret, subopts_newt, lw=lw, color="crimson", label=ours)

    ax.set_xlabel("Seconds", fontsize=fontsize)
    ax.set_ylabel("Suboptimality", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlim([0, xmax])
    ax.set_ylim([ymin, ymax])
    ax.legend(loc=loc, fontsize=fontsize)
    fig.savefig("subopts_" + name + ".pdf", bbox_inches="tight")

    # Make solution error plot.
    lw = 3
    fontsize = 14
    loc = "upper right"

    fig,ax = plt.subplots()
    ax.semilogy(solve_times_SCS_ret, errors_SCS_ret, lw=lw, color="blue", label="SCS")
    ax.semilogy(solve_times_newt_ret, errors_newt_ret, lw=lw, color="crimson", label=ours)

    ax.set_xlabel("Seconds", fontsize=fontsize)
    ax.set_ylabel("Estimation error", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlim([0, xmax])
    ax.set_ylim([ymin, ymax])
    ax.legend(loc=loc, fontsize=fontsize)
    fig.savefig("estims_" + name + ".pdf", bbox_inches="tight")