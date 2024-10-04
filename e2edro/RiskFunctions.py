import cvxpy as cp
import numpy as np

# Risk functions module
#
# This module defines the financial risk measures to be used in the optimization layer of the E2E
# problem.
#


def p_var(z: cp.Expression, c: float, x: np.ndarray) -> cp.Expression:
    """
    Compute the squared error for the given input.

    :param z: A cvxpy expression (decision variable)
    :param c: A constant threshold or target value
    :param x: A numpy array (weights or features)
    :return: The squared error expression
    """
    return cp.square(x @ z - c)


def p_mad(z: cp.Expression, c: float, x: np.ndarray) -> cp.Expression:
    """
    Compute the mean absolute deviation for the given input.

    :param z: A cvxpy expression (decision variable)
    :param c: A constant threshold or target value
    :param x: A numpy array (weights or features)
    :return: The absolute deviation expression
    """
    return cp.abs(x @ z - c)


# Define test data
z = cp.Variable(3)  # Decision variable (portfolio weights)
c = 0.02  # Centering parameter (expected return)
x = np.array(
    [
        [0.05, 0.02, -0.01],  # Realized returns for multiple scenarios
        [0.03, -0.01, 0.04],
        [-0.02, 0.01, 0.01],
    ]
)
if __name__ == "__main__":

    # Test variance function (p_var)
    print("\nTesting p_var...")
    var_expr = p_var(z, c, x[0])  # Apply p_var to the first row of x
    objective_var = cp.Minimize(var_expr)  # Minimize the variance
    constraints = [
        cp.sum(z) == 1,
        z >= 0,
    ]  # Portfolio constraints: sum of weights = 1, weights >= 0
    problem_var = cp.Problem(objective_var, constraints)
    var_opt_value = problem_var.solve()

    # Output results for variance minimization
    print("Optimized portfolio weights (Variance):", z.value)
    print("Variance objective value:", var_opt_value)

    # Reinitialize decision variable for MAD problem
    z = cp.Variable(3)

    # Test MAD function (p_mad)
    print("\nTesting p_mad...")
    mad_expr = p_mad(z, c, x[0])  # Apply p_mad to the first row of x
    objective_mad = cp.Minimize(mad_expr)  # Minimize the MAD
    problem_mad = cp.Problem(objective_mad, constraints)
    mad_opt_value = problem_mad.solve()

    # Output results for MAD minimization
    print("Optimized portfolio weights (MAD):", z.value)
    print("MAD objective value:", mad_opt_value)
