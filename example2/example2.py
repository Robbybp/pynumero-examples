"""An example where we plot iterate vectors (projected onto two dimensions) during a solve
using an intermediate callback

NOTE: This example depends on the pglib-opf dataset, which can be downloaded here:

    https://github.com/power-grid-lib/pglib-opf

"""

import os
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from egret.parsers.matpower_parser import create_ModelData
from egret.models.acopf import create_psv_acopf_model
import matplotlib.pyplot as plt

FILEDIR = os.path.dirname(__file__)
PGLIB_DIR = os.path.join(FILEDIR, "pglib-opf")
fname = "pglib_opf_case118_ieee.m"
fpath = os.path.join(PGLIB_DIR, fname)
data = create_ModelData(fpath)

generator_lookup = dict(data.elements("generator"))
online_generators = [i for i, g in data.elements("generator") if g["in_service"]]
sorted_generators = sorted(online_generators, key=lambda i: generator_lookup[i]["p_max"], reverse=True)

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
plotgens = sorted_generators[:2]
plotvars = [m.pg[i] for i in plotgens]
indices = nlp.get_primal_indices(plotvars)

plotdata = [
    [data["primals"][i] for data in callback.iterate_data]
    for i in indices
]
x, y = plotvars
xdata, ydata = plotdata

# Plot state trajectories
fig, ax = plt.subplots()
ax.plot(xdata, ydata)
ax.scatter([xdata[0]], [ydata[0]], marker=".", s=100, label="Initial", color="black", zorder=10)
ax.scatter([xdata[-1]], [ydata[-1]], marker="*", s=100, label="Initial", color="black", zorder=10)
ax.plot([x.lb, x.ub], [y.lb, y.lb], color="red")
ax.plot([x.lb, x.ub], [y.ub, y.ub], color="red")
ax.plot([x.lb, x.lb], [y.lb, y.ub], color="red")
ax.plot([x.ub, x.ub], [y.lb, y.ub], color="red")
ax.set_ylim(-0.5, 6)
ax.set_xlim(4, 10)
plt.show()

# Now we want to plot contours of the objective.
# To a second-order approximation, the objective is the quadratic function
# defined by the reduced Hessian.
