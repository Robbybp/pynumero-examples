"""An example of checking the eigenvalues of the reduced Hessian
"""

import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from svi.auto_thermal_reformer.fullspace_flowsheet import make_optimization_model
import numpy as np
import scipy as sp


def make_and_solve_model(**kwds):
    # First, we make our model
    P = kwds.pop("P", 1.5e6)
    # With X = 0.95, we converge. With X = 0.94, we don't. Go figure
    X = kwds.pop("X", 0.94)
    m = make_optimization_model(X, P)
    m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    # Does the model solve?
    solver = pyo.SolverFactory("cyipopt", options=kwds)
    res = solver.solve(m, tee=True)
    # Converges infeasible in 900 iter
    return m, res


#make_and_solve_model()
#m, res = make_and_solve_model(max_iter=43)
m, res = make_and_solve_model(X=0.95)
nlp = PyomoNLP(m)

# Why could we have a regularization coefficient?
# - Equality Jacobian is not full row rank
# - Reduced Hessian is not positive definite

# You can use the model or nlp here. If we already have the nlp, it's somewhat
# more efficient to use it.
igraph = IncidenceGraphInterface(nlp, include_inequality=False)
vdm, cdm = igraph.dulmage_mendelsohn()
assert not cdm.unmatched
# This proves that we are not structurally singular.
# Can we prove that we're numerically full row rank?
# Easiest way (for me) is to get degrees of freedom.
# Assume we know what these are.
dof_varnames = [
    "fs.reformer_bypass.split_fraction[0.0,bypass_outlet]",
    "fs.reformer_mix.steam_inlet_state[0.0].flow_mol",
    "fs.feed.properties[0.0].flow_mol",
]
dofvars = [m.find_component(name) for name in dof_varnames] 
assert not any(v is None for v in dofvars)
dofvar_set = ComponentSet(dofvars)
basicvars = list(filter(lambda x: x not in dofvar_set, nlp.get_pyomo_variables()))
print(f"N. dof variables:   {len(dofvars)}")
print(f"N. basic variables: {len(basicvars)}")

eqcons = nlp.get_pyomo_equality_constraints()
basic_submatrix = nlp.extract_submatrix_jacobian(basicvars, eqcons)
# This throws an error if the matrix is singular
try:
    lu = sp.sparse.linalg.splu(basic_submatrix)
    print(f"Eq. Jac. has rank at least {len(basicvars)}")
except RuntimeError:
    print("Eq. Jac. is not (necessarily) full row rank")
# This proves that our equality Jacobian is full row rank

# Now we can start worrying about the reduced Hessian
# First, we want to make sure we know our Lagrangian convention
def get_gradient_of_lagrangian(
    nlp,
    primal_lb_multipliers,
    primal_ub_multipliers,
):
    # PyNumero NLPs contain constraint multipliers, but does not define a convention.
    # We still need:
    # - primal LB/UB multipliers
    # We should not need slack multipliers (Ipopt should take care of this...)
    grad_obj = nlp.evaluate_grad_objective()

    # There is no way this works. We will probably need to separate equality and
    # inequality multipliers.
    jac = nlp.evaluate_jacobian()
    duals = nlp.get_duals()
    # Each constraint gradient times its multiplier
    conjac_term = jac.transpose().dot(duals)

    grad_lag = (
        - grad_obj
        - conjac_term
        + primal_lb_multipliers
        - primal_ub_multipliers
    )
    return grad_lag


def test_gradient_of_lagrangian():
    m = pyo.ConcreteModel()
    m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    m.x = pyo.Var([1,2,3], initialize=1.0, bounds=(0, 10))
    m.eq = pyo.Constraint(pyo.PositiveIntegers)
    m.ineq = pyo.Constraint(pyo.PositiveIntegers)
    m.eq[1] = m.x[1] * m.x[2]**1.5 == m.x[3]
    m.eq[2] = m.x[1] + 2*m.x[2] + 3*m.x[3] == 5
    m.ineq[1] = m.x[1] >= 2
    m.ineq[2] = m.x[3] <= 4
    m.obj = pyo.Objective(expr=m.x[1]**2 + 2*m.x[2]**2 + 3*m.x[3]**2)
    solver = pyo.SolverFactory("cyipopt")
    res = solver.solve(m, tee=False)
    pyo.assert_optimal_termination(res)
    nlp = PyomoNLP(m)
    lbmult = [m.ipopt_zL_out[v] for v in nlp.get_pyomo_variables()]
    ubmult = [m.ipopt_zU_out[v] for v in nlp.get_pyomo_variables()]
    grad_lag = get_gradient_of_lagrangian(nlp, lbmult, ubmult)
    assert all(abs(v) < 1e-8 for v in grad_lag)


test_gradient_of_lagrangian()
# Okay we got our Lagrangian convention right. Now we can construct the reduced Hessian.
# The reduced Hessian is the Hessian of the Lagrangian projected onto the null
# space of the equality constraint Jacobian.
# So first we construct the Hessian of the Lagrangian.
# We'll construct with variables in the order (dof_vars, other_vars).

varorder = dofvars + basicvars
varindices = nlp.get_primal_indices(varorder)
hess = nlp.extract_submatrix_hessian_lag(varorder, varorder)
# This is the Hessian of the Lagrangian, but it's missing the terms for bounds
# and inequalities.
# We'll start with the bound term, which is e.g., mult / (ub - var)

lbmult = sp.sparse.diags([m.ipopt_zL_out[v] for v in nlp.get_pyomo_variables()])
ubmult = sp.sparse.diags([m.ipopt_zU_out[v] for v in nlp.get_pyomo_variables()])

primal_val = nlp.get_primals()[varindices]
primal_lb = nlp.primals_lb()[varindices]
primal_ub = nlp.primals_ub()[varindices]
# If bounds are exactly active, we will divide by zero, so we apply a small perturbation
active_lbs = np.where(primal_lb == primal_val)[0]
active_ubs = np.where(primal_ub == primal_val)[0]
# Push values into interior
primal_val[active_lbs] += 1e-8
primal_val[active_ubs] -= 1e-8
primal_ubslack_inv = sp.sparse.diags(1 / (primal_ub - primal_val))
primal_lbslack_inv = sp.sparse.diags(1 / (primal_val - primal_lb))
bound_term = lbmult @ primal_ubslack_inv + ubmult @ primal_lbslack_inv

# Now we construct the inequality term
ineqcons = nlp.get_pyomo_inequality_constraints()
ineq_jac = nlp.extract_submatrix_jacobian(varorder, ineqcons)
ineq_val = nlp.evaluate_ineq_constraints()
ineq_val_inv = sp.sparse.diags(1 / ineq_val)
ineq_duals = sp.sparse.diags(nlp.get_duals_ineq())
ineq_term = ineq_jac.transpose() @ ineq_val_inv @ ineq_duals @ ineq_jac

# Now we can add these terms to our Hessian
hess += bound_term
hess -= ineq_term

# Now we can project onto the null space of the eq. Jac.
dof_submatrix = nlp.extract_submatrix_jacobian(dofvars, eqcons)
ndof = len(dofvars)
# Construct a "fundamental null basis" by pivoting on our basic submatrix
# Construct this as a dense matrix. Our backsolve below will give us a dense
# matrix anyway.
dof_nullbasis = np.identity(ndof)
# We've done this above already...
lu = sp.sparse.linalg.splu(basic_submatrix)
basic_nullbasis = - lu.solve(dof_submatrix.toarray())
nullbasis = np.vstack((dof_nullbasis, basic_nullbasis))

rh = nullbasis.transpose() @ hess @ nullbasis
print(rh)
evals, evecs = np.linalg.eig(rh)
print(evals)
