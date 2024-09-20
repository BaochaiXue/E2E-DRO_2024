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
