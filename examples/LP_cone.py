import cvxpy as cp
import numpy as np
np.random.seed(0)

from newton_admm import newton_admm, problems
from plot import plotter

# Benchmark LP
name = "LP"
m,n = 300,600
beta, prob, data = problems.lp(m,n)
prob.solve(solver="ECOS", abstol=1e-14, reltol=1e-14, feastol=1e-14, 
           verbose=True, max_iters=50)
betastar = np.array(beta.value).flatten()
fstar = prob.value

beta_from_x = data['beta_from_x']
newton_out = newton_admm(data, data['dims'], benchmark=(beta_from_x, betastar), 
                         maxiters=400, verbose=1)
scs_out = newton_admm(data, data['dims'], benchmark=(beta_from_x, betastar), 
                         admm_maxiters=20000, maxiters=0, verbose=1) 

plotter(newton_out, scs_out, fstar, name, xmax=49, ymin=1e-12, ymax=1e2)