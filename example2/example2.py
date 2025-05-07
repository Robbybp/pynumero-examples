"""An example where we plot iterate vectors (projected onto two dimensions) during a solve
using an intermediate callback

NOTE: This example depends on the pglib-opf dataset, which can be downloaded here:

    https://github.com/power-grid-lib/pglib-opf

"""

import os
import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from egret.parsers.matpower_parser import create_ModelData
from egret.models.acopf import create_psv_acopf_model
import matplotlib.pyplot as plt
import matplotlib

FILEDIR = os.path.dirname(__file__)
PGLIB_DIR = os.path.join(FILEDIR, "pglib-opf")
fname = "pglib_opf_case118_ieee.m"
fpath = os.path.join(PGLIB_DIR, fname)
data = create_ModelData(fpath)

generator_lookup = dict(data.elements("generator"))
bus_lookup = dict(data.elements("bus"))
online_generators = [i for i, g in data.elements("generator") if g["in_service"]]
sorted_generators = sorted(online_generators, key=lambda i: generator_lookup[i]["p_max"], reverse=True)
nonreference_generators = [
    i for i in sorted_generators
    if bus_lookup[generator_lookup[i]["bus"]]["matpower_bustype"] != "ref"
]

# The second argument here is ModelData, which I already have. (Maybe it's been altered?)
m, _ = create_psv_acopf_model(data)

solver = pyo.SolverFactory("cyipopt")
# Does the model solve?
res = solver.solve(m, tee=True)
pyo.assert_optimal_termination(res)

# Now we want to track the state during the solve. Let's create a fresh model
m, _ = create_psv_acopf_model(data)
m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

# We'll track the state with a callback object
class Callback:

    def __init__(self):
        self.iterate_data = []

    def __call__(
        self,
        nlp,
        ipopt_problem,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        """Intermediate callback in the Pyomo CyIpopt interface is a 13-argument
        function
        """
        iterate = ipopt_problem.get_current_iterate(scaled=False)
        # See here: https://cyipopt.readthedocs.io/en/stable/reference.html
        # for the keys of the dict that this function returns.
        primals = iterate["x"]
        duals = iterate["mult_g"]
        lbmult = iterate["mult_x_L"]
        ubmult = iterate["mult_x_U"]
        data = dict(primals=primals, duals=duals, lbmult=lbmult, ubmult=ubmult)
        self.iterate_data.append(data)


callback = Callback()

solver.config.intermediate_callback = callback
solver.solve(m, tee=True)

# We were a bit fast and loose with the variable order above, so let's sanity
# check our last iterate.
nlp = PyomoNLP(m)
# These could be slightly different due to Ipopt's honor_original_bounds option,
# or other shenanigans
assert all(nlp.get_primals() == callback.iterate_data[-1]["primals"])

# Identify the variables we want to plot
# We'll plot the power on the two largest non-reference generators
#plotgens = nonreference_generators[:2]
# I chose these generators to make sure we have positive eigenvalues in these
# coordinates. This is kind of cheating. I'm not really sure why the eigenvalue
# in the "g0" coordinate of the reduced Hessian is negative...
g1 = nonreference_generators[2]
g2 = nonreference_generators[3]
plotvars = [m.pg[g1], m.pg[g2]]
indices = nlp.get_primal_indices(plotvars)

plotdata = [
    [data["primals"][i] for data in callback.iterate_data]
    for i in indices
]
x, y = plotvars
xdata, ydata = plotdata

# Plot state trajectories
fig, ax = plt.subplots()
ax.plot([x.lb, x.ub], [y.lb, y.lb], color="red", label="Bounds")
ax.plot([x.lb, x.ub], [y.ub, y.ub], color="red")
ax.plot([x.lb, x.lb], [y.lb, y.ub], color="red")
ax.plot([x.ub, x.ub], [y.lb, y.ub], color="red")
ax.plot(xdata, ydata)
ax.scatter([xdata[0]], [ydata[0]], marker=".", s=100, label="Initial", color="black", zorder=10)
ax.scatter([xdata[-1]], [ydata[-1]], marker="*", s=100, label="Optimal", color="black", zorder=10)
ax.legend()
#plt.show()

# Now we want to plot contours of the objective.
# To a second-order approximation, the objective is the quadratic function
# defined by the reduced Hessian.
from reduced_hessian import get_reduced_hessian, project_onto, get_gradient_of_lagrangian
lbmult = [m.ipopt_zL_out[x] for x in nlp.get_pyomo_variables()]
ubmult = [m.ipopt_zU_out[x] for x in nlp.get_pyomo_variables()]
gradlag = get_gradient_of_lagrangian(nlp, lbmult, ubmult)
assert all(abs(gradlag) <= 1e-7)

dofvars = [m.pg[i] for i in nonreference_generators] + list(m.qg.values())
dofvar_map = ComponentMap((v, i) for i, v in enumerate(dofvars))
basicvars = [v for v in nlp.get_pyomo_variables() if v not in dofvar_map]
eqcons = nlp.get_pyomo_equality_constraints()
basic_submatrix = nlp.extract_submatrix_jacobian(basicvars, eqcons)
print(basic_submatrix)
import scipy as sp
import numpy as np
lu = sp.sparse.linalg.splu(basic_submatrix)
cond = np.linalg.cond(basic_submatrix.toarray())
print(f"Condition number of basic submatrix: {cond:1.2E}")
# I am sufficiently convinced that these degrees of freedom are valid. Now we
# can get the reduced Hessian.

rh = get_reduced_hessian(nlp, dofvars, lbmult, ubmult)
# `x` and `y` are the variables we're plotting.
plot_indices = [dofvar_map[x], dofvar_map[y]]
projected_rh = project_onto(rh, plot_indices)
print(np.linalg.eig(projected_rh))
evals, evecs = np.linalg.eig(projected_rh)
assert all(evals > 0)

# Center our contours on the optimal termination point
center = (xdata[-1], ydata[-1])
angle = np.rad2deg(np.arctan2(evecs[1,0], evecs[1,1]))
levels = [1e0, 1e1, 1e2, 1e3]
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for l in levels:
    width = 2 * (l / evals[0])**0.5
    height = 2 * (l / evals[1])**0.5
    ellipse = matplotlib.patches.Ellipse(
        center,
        width=width,
        height=height,
        angle=angle,
        edgecolor=tuple([0.7]*3),
        facecolor="none",
    )
    ax.add_patch(ellipse)
# Reset plot limits so these aren't influenced by ellipse
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.show()
