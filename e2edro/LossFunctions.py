# Financial performance loss functions for E2E learning framework
#
####################################################################################################
## Import libraries
####################################################################################################
import torch


####################################################################################################
# Performance loss functions
####################################################################################################
def _prepare_batch(z_star: torch.Tensor, y_perf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure ``z_star`` and ``y_perf`` both have a batch dimension."""
    if z_star.dim() == 1:
        z_star = z_star.unsqueeze(0)
    if z_star.dim() == 2 and z_star.size(-1) != 1:
        z_star = z_star.unsqueeze(-1)

    if y_perf.dim() == 2:
        y_perf = y_perf.unsqueeze(0)
    return z_star.squeeze(-1), y_perf


def single_period_loss(z_star: torch.Tensor, y_perf: torch.Tensor) -> torch.Tensor:
    """Loss based on the out-of-sample portfolio return for the next time step.

    ``z_star`` and ``y_perf`` can contain an optional batch dimension which will
    be averaged over.
    """

    z_star, y_perf = _prepare_batch(z_star, y_perf)
    ret = (y_perf[:, 0, :] * z_star).sum(dim=1)
    return -ret.mean()


def single_period_over_var_loss(z_star: torch.Tensor, y_perf: torch.Tensor) -> torch.Tensor:
    """Loss based on the next period return scaled by realized volatility."""

    z_star, y_perf = _prepare_batch(z_star, y_perf)
    rets = torch.matmul(y_perf, z_star.unsqueeze(-1)).squeeze(-1)
    loss = -rets[:, 0] / rets.std(dim=1)
    return loss.mean()


def sharpe_loss(z_star: torch.Tensor, y_perf: torch.Tensor) -> torch.Tensor:
    """Loss function based on the out-of-sample Sharpe ratio."""

    z_star, y_perf = _prepare_batch(z_star, y_perf)
    rets = torch.matmul(y_perf, z_star.unsqueeze(-1)).squeeze(-1)
    loss = -rets.mean(dim=1) / rets.std(dim=1)
    return loss.mean()
