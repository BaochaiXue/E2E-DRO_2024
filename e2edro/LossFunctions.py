import torch
from torch import Tensor


# Performance loss functions with type hints and improved comments
def single_period_loss(z_star: Tensor, y_perf: Tensor) -> Tensor:
    """
    Calculate the single-period loss based on the out-of-sample portfolio return.

    This function computes the out-of-sample portfolio return for a given portfolio over the next
    time step. It computes the loss as the negative return since optimization typically focuses
    on minimizing the loss, and maximizing returns translates into minimizing negative returns.

    :param z_star: Tensor of shape (n_y, 1) representing the optimal portfolio weights.
    :param y_perf: Tensor of shape (perf_period, n_y) representing the realized returns.
    :return: A scalar tensor representing the realized return at the first time step (negative).
    """
    # Calculate the portfolio return for the first time step and negate it (since we want to minimize loss)
    return -y_perf[0] @ z_star


def single_period_over_var_loss(z_star: Tensor, y_perf: Tensor) -> Tensor:
    """
    Calculate the loss as the portfolio return divided by the portfolio's volatility.

    This function computes the portfolio return at the first time step and divides it by the
    realized volatility (standard deviation) of the portfolio returns over the performance period.
    This provides a return-over-risk measure, which is often used in portfolio analysis.

    :param z_star: Tensor of shape (n_y, 1) representing the optimal portfolio weights.
    :param y_perf: Tensor of shape (perf_period, n_y) representing the realized returns.
    :return: A scalar tensor representing the return over realized volatility (negative).
    """
    # Calculate the portfolio returns over the entire performance period
    portfolio_returns = y_perf @ z_star
    # Calculate the standard deviation (volatility) of the portfolio returns, adding epsilon for numerical stability
    volatility = torch.std(portfolio_returns, unbiased=True) + 1e-6
    # Calculate the return at the first time step and divide by the volatility, then negate for loss
    return -portfolio_returns[0] / volatility


def sharpe_loss(z_star: Tensor, y_perf: Tensor) -> Tensor:
    """
    Calculate the loss based on the Sharpe ratio over a performance period.

    This function computes a simplified Sharpe ratio, which is the ratio of the mean portfolio
    return to its standard deviation (volatility) over the performance period. The loss is defined
    as the negative Sharpe ratio to allow for minimization.

    :param z_star: Tensor of shape (n_y, 1) representing the optimal portfolio weights.
    :param y_perf: Tensor of shape (perf_period, n_y) representing the realized returns.
    :return: A scalar tensor representing the negative Sharpe ratio.
    """
    # Calculate the portfolio returns over the entire performance period
    portfolio_returns = y_perf @ z_star
    # Calculate the mean return of the portfolio
    mean_return = torch.mean(portfolio_returns)
    # Calculate the standard deviation (volatility) of the portfolio returns, adding epsilon for numerical stability
    volatility = torch.std(portfolio_returns, unbiased=True) + 1e-6
    # Calculate the Sharpe ratio and negate it for loss
    return -mean_return / volatility


if __name__ == "__main__":
    # Example portfolio weights (3 assets)
    z_star = torch.tensor([0.3, 0.5, 0.2])
    # Realized returns for 3 assets over 3 periods
    y_perf = torch.tensor(
        [
            [0.01, 0.02, -0.01],
            [0.03, -0.01, 0.04],
            [0.02, 0.01, 0.01],
        ]
    )

    # Test the single-period loss function
    print("Testing single_period_loss...")
    loss_sp = single_period_loss(z_star, y_perf)
    print(f"Single period loss: {loss_sp.item()}")

    # Test the single-period-over-volatility loss function
    print("\nTesting single_period_over_var_loss...")
    loss_sp_var = single_period_over_var_loss(z_star, y_perf)
    print(f"Single period loss over volatility: {loss_sp_var.item()}")

    # Test the Sharpe ratio loss function
    print("\nTesting sharpe_loss...")
    loss_sharpe = sharpe_loss(z_star, y_perf)
    print(f"Sharpe ratio loss: {loss_sharpe.item()}")
