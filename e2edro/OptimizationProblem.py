import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import e2edro.RiskFunctions as rf
import e2edro.LossFunctions as lf
import e2edro.PortfolioClasses as pc
import e2edro.DataLoad as dl
from collections.abc import Callable
import psutil
from e2edro.RiskFunctions import p_var, p_mad


num_cores: int = psutil.cpu_count()
torch.set_num_threads(num_cores)
if psutil.MACOS:
    num_cores = 0

if __name__ == "__main__":
    print(f"Number of cores: {num_cores}")

# check if we can use cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    print(f"Device: {device}")


####################################################################################################
# CvxpyLayers: Differentiable optimization layers (nominal and distributionally robust)
####################################################################################################
# ---------------------------------------------------------------------------------------------------
# base_mod: CvxpyLayer that declares the portfolio optimization problem
# ---------------------------------------------------------------------------------------------------


def base_mod(n_y: int, n_obs: int, prisk: Callable) -> CvxpyLayer:
    """
    Base optimization problem declared as a CvxpyLayer object.

    :param n_y: Number of assets.
    :param n_obs: Number of scenarios in the dataset.
    :param prisk: Portfolio risk function.
    :return: CvxpyLayer representing the optimization layer.
    """
    # Variables
    z: cp.Variable = cp.Variable(
        (n_y, 1), nonneg=True
    )  # Portfolio weights (long-only positions)

    # Parameters
    y_hat: cp.Parameter = cp.Parameter(
        n_y
    )  # Predicted outcomes (e.g., expected returns)

    # Constraints
    constraints: list[cp.Constraint] = [
        cp.sum(z) == 1  # Budget constraint: sum of weights equals 1
    ]

    # Objective function
    objective: cp.Minimize = cp.Minimize(
        -y_hat @ z
    )  # Maximize returns by minimizing negative expected returns

    # Construct optimization problem and differentiable layer
    problem: cp.Problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[y_hat], variables=[z])


# ---------------------------------------------------------------------------------------------------
# nominal: CvxpyLayer that declares the nominal portfolio optimization problem
# ---------------------------------------------------------------------------------------------------
def nominal(n_y: int, n_obs: int, prisk: Callable) -> CvxpyLayer:
    """
    Nominal optimization problem declared as a CvxpyLayer object.

    :param n_y: Number of assets.
    :param n_obs: Number of scenarios in the dataset.
    :param prisk: Portfolio risk function.
    :return: CvxpyLayer representing the optimization layer.
    """
    # Variables
    z: cp.Variable = cp.Variable(
        (n_y, 1), nonneg=True
    )  # Portfolio weights (long-only positions)
    c_aux: cp.Variable = cp.Variable()  # Auxiliary variable for risk calculation
    obj_aux: cp.Variable = cp.Variable(
        n_obs
    )  # Objective auxiliary variable for each scenario
    mu_aux: cp.Variable = cp.Variable()  # Expected portfolio return

    # Parameters
    ep: cp.Parameter = cp.Parameter((n_obs, n_y))  # Scenario matrix of residuals
    y_hat: cp.Parameter = cp.Parameter(
        n_y
    )  # Predicted outcomes (e.g., expected returns)
    gamma: cp.Parameter = cp.Parameter(nonneg=True)  # Risk aversion coefficient

    # Constraints
    constraints: list[cp.Constraint] = [
        cp.sum(z) == 1,  # Budget constraint: sum of weights equals 1
        mu_aux == y_hat @ z,  # Calculate expected return
    ]
    for i in range(n_obs):
        constraints.append(
            obj_aux[i] >= prisk(z, c_aux, ep[i])
        )  # Add risk constraints for each scenario

    # Objective function
    objective: cp.Minimize = cp.Minimize(
        (1 / n_obs) * cp.sum(obj_aux) - gamma * mu_aux
    )  # Minimize risk-adjusted return

    # Construct optimization problem and differentiable layer
    problem: cp.Problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[ep, y_hat, gamma], variables=[z])


# ---------------------------------------------------------------------------------------------------
# Total Variation: sum_t abs(p_t - q_t) <= delta
# ---------------------------------------------------------------------------------------------------
def tv(
    n_y: int,
    n_obs: int,
    prisk: Callable[[cp.Variable, cp.Variable, np.ndarray], cp.Expression],
) -> CvxpyLayer:
    """
    Declares a DRO optimization problem using 'Total Variation' distance to define the probability
    ambiguity set, based on Ben-Tal et al. (2013).

    :param n_y: Number of assets
    :param n_obs: Number of scenarios in the dataset
    :param prisk: Callable that defines the portfolio risk function

    :return: CvxpyLayer representing the differentiable optimization layer
    """
    # Decision Variables
    z: cp.Variable = cp.Variable((n_y, 1), nonneg=True)  # Portfolio weights (long-only)
    c_aux: cp.Variable = (
        cp.Variable()
    )  # Auxiliary scalar variable for variance linearization
    lambda_aux: cp.Variable = cp.Variable(nonneg=True)  # Scalar for DR counterpart
    eta_aux: cp.Variable = cp.Variable()  # Scalar for tractable DR counterpart
    beta_aux: cp.Variable = cp.Variable(n_obs)  # Auxiliary vector for DR counterpart
    mu_aux: cp.Variable = cp.Variable()  # Scalar for conditional expected return

    # Parameters
    ep: cp.Parameter = cp.Parameter((n_obs, n_y))  # Residuals matrix (n_obs x n_y)
    y_hat: cp.Parameter = cp.Parameter(
        n_y
    )  # Predicted outcomes (conditional expected returns)
    gamma: cp.Parameter = cp.Parameter(
        nonneg=True
    )  # Trade-off between return and error
    delta: cp.Parameter = cp.Parameter(
        nonneg=True
    )  # Maximum allowed distance (TV constraint)

    # Constraints
    constraints: list[cp.Constraint] = [
        cp.sum(z) == 1,  # Budget constraint: weights sum to 1 (100%)
        beta_aux
        >= -lambda_aux,  # Ensure beta_aux is greater than or equal to -lambda_aux
        mu_aux == y_hat @ z,  # Calculate the conditional expected return
    ]

    # Add risk constraints for each scenario
    for i in range(n_obs):
        constraints.append(
            beta_aux[i] >= prisk(z, c_aux, ep[i]) - eta_aux
        )  # TV risk constraints
        constraints.append(
            lambda_aux >= prisk(z, c_aux, ep[i]) - eta_aux
        )  # TV auxiliary constraint

    # Objective function: Minimize the total variation distance-based risk-adjusted return
    objective: cp.Minimize = cp.Minimize(
        eta_aux + delta * lambda_aux + (1 / n_obs) * cp.sum(beta_aux) - gamma * mu_aux
    )

    # Construct and return the optimization problem as a CvxpyLayer
    problem: cp.Problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta], variables=[z])


# ---------------------------------------------------------------------------------------------------
# Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta
# ---------------------------------------------------------------------------------------------------


def hellinger(
    n_y: int,
    n_obs: int,
    prisk: Callable[[cp.Variable, cp.Variable, np.ndarray], cp.Expression],
) -> CvxpyLayer:
    """
    Declares a DRO optimization problem using the Hellinger distance to define the probability
    ambiguity set, based on Ben-Tal et al. (2013).

    :param n_y: Number of assets
    :param n_obs: Number of scenarios in the dataset
    :param prisk: Callable that defines the portfolio risk function

    :return: CvxpyLayer representing the differentiable optimization layer
    """
    # Decision Variables
    z: cp.Variable = cp.Variable((n_y, 1), nonneg=True)  # Portfolio weights (long-only)
    c_aux: cp.Variable = cp.Variable()  # Auxiliary scalar for variance linearization
    lambda_aux: cp.Variable = cp.Variable(nonneg=True)  # Scalar for DR counterpart
    xi_aux: cp.Variable = cp.Variable()  # Scalar for tractable DR counterpart
    beta_aux: cp.Variable = cp.Variable(
        n_obs, nonneg=True
    )  # Auxiliary vector for DR counterpart
    tau_aux: cp.Variable = cp.Variable(
        n_obs, nonneg=True
    )  # Auxiliary vector for SOC constraint
    mu_aux: cp.Variable = cp.Variable()  # Scalar for conditional expected return

    # Parameters
    ep: cp.Parameter = cp.Parameter((n_obs, n_y))  # Residuals matrix (n_obs x n_y)
    y_hat: cp.Parameter = cp.Parameter(
        n_y
    )  # Predicted outcomes (conditional expected returns)
    gamma: cp.Parameter = cp.Parameter(
        nonneg=True
    )  # Trade-off between return and error
    delta: cp.Parameter = cp.Parameter(
        nonneg=True
    )  # Maximum allowed Hellinger distance

    # Constraints
    constraints: list[cp.Constraint] = [
        cp.sum(z) == 1,  # Budget constraint: weights sum to 1 (100%)
        mu_aux == y_hat @ z,  # Calculate the conditional expected return
    ]

    # Add constraints for each scenario based on the Hellinger distance formulation
    for i in range(n_obs):
        constraints.append(xi_aux + lambda_aux >= prisk(z, c_aux, ep[i]) + tau_aux[i])
        constraints.append(beta_aux[i] >= cp.quad_over_lin(lambda_aux, tau_aux[i]))

    # Objective function: Minimize risk-adjusted return based on Hellinger distance
    objective: cp.Minimize = cp.Minimize(
        xi_aux
        + (delta - 1) * lambda_aux
        + (1 / n_obs) * cp.sum(beta_aux)
        - gamma * mu_aux
    )

    # Construct and return the optimization problem as a CvxpyLayer
    problem: cp.Problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta], variables=[z])


if __name__ == "__main__":
    # Example usage of the base optimization problem
    base_layer: CvxpyLayer = base_mod(3, 10, rf.p_var)
    y_hat: torch.Tensor = torch.randn(3)
    z_star: torch.Tensor = base_layer(y_hat)[0]
    print(f"Optimal portfolio weights (base): {z_star}")

    # Example usage of the nominal optimization problem
    nominal_layer: CvxpyLayer = nominal(3, 10, rf.p_var)
    y_hat: torch.Tensor = torch.randn(3)
    ep: torch.Tensor = torch.randn(10, 3)
    gamma: torch.Tensor = torch.tensor(0.1)
    z_star: torch.Tensor = nominal_layer(ep, y_hat, gamma)[0]
    print(f"Optimal portfolio weights (nominal): {z_star}")

    # Example usage of the TV optimization problem
    tv_layer: CvxpyLayer = tv(3, 10, rf.p_var)
    y_hat: torch.Tensor = torch.randn(3)
    ep: torch.Tensor = torch.randn(10, 3)
    gamma: torch.Tensor = torch.tensor(0.1)
    delta: torch.Tensor = torch.tensor(0.1)
    z_star: torch.Tensor = tv_layer(ep, y_hat, gamma, delta)[0]
    print(f"Optimal portfolio weights (TV): {z_star}")

    # Example usage of the Hellinger optimization problem
    hellinger_layer: CvxpyLayer = hellinger(3, 10, rf.p_var)
    y_hat: torch.Tensor = torch.randn(3)
    ep: torch.Tensor = torch.randn(10, 3)
    gamma: torch.Tensor = torch.tensor(0.1)
    delta: torch.Tensor = torch.tensor(0.1)
    z_star: torch.Tensor = hellinger_layer(ep, y_hat, gamma, delta)[0]
    print(f"Optimal portfolio weights (Hellinger): {z_star}")
