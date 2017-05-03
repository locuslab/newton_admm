import cvxpy as cp
import numpy as np
np.random.seed(0)

from newton_admm import newton_admm, problems
from plot import plotter

# Benchmark LP
name = "PO"
p = 2500
beta, prob, data = problems.portfolio_opt(p)
beta_from_x = data['beta_from_x']
# ECOS cannot solve this to sufficient precision!
# prob.solve(solver="ECOS", abstol=1e-14, reltol=1e-14, feastol=1e-14, 
#            verbose=True)
# prob.solve(solver="SCS", eps=1e-14)
# betastar = np.array(beta.value).flatten()
# fstar = prob.value

baseline_out = newton_admm(data, data['dims'], maxiters=100, res_tol = 1e-14, verbose=10)
betastar = beta_from_x(baseline_out['x'])
fstar = baseline_out['info']['fstar']

newton_out = newton_admm(data, data['dims'], benchmark=(beta_from_x, betastar), 
                         maxiters=100, ridge=1e-4, verbose=1)
scs_out = newton_admm(data, data['dims'], benchmark=(beta_from_x, betastar), 
                         admm_maxiters=2000, maxiters=0, verbose=100) 

plotter(newton_out, scs_out, fstar, name, xmax=100, ymin=1e-12, ymax=1e2) 