# Naive Model Module
#
####################################################################################################
## Import libraries
####################################################################################################
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import RiskFunctions as rf
from . import PortfolioClasses as pc

# ``e2edro`` pulls in ``cvxpylayers`` which depends on the optional
# ``diffcp`` package. Importing it unconditionally prevents using the
# simpler models (e.g. ``equal_weight``) on systems where ``diffcp`` is
# not installed.  To allow these models to run without the optional
# dependency we defer importing ``e2edro`` until it is actually needed
# (in ``pred_then_opt``).



####################################################################################################
# Naive 'predict-then-optimize'
####################################################################################################
class pred_then_opt(nn.Module):
    """Naive 'predict-then-optimize' portfolio construction module"""

    def __init__(
        self, n_x, n_y, n_obs, set_seed=None, prisk="p_var", opt_layer="nominal"
    ):
        """Naive 'predict-then-optimize' portfolio construction module

        This NN module implements a linear prediction layer 'pred_layer' and an optimization layer
        'opt_layer'. The model is 'naive' since it optimizes each layer separately.

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: String. Portfolio risk function. Used in the opt_layer

        Output
        pred_then_opt: nn.Module object
        """
        super(pred_then_opt, self).__init__()

        if set_seed is not None:
            torch.manual_seed(set_seed)
            self.seed = set_seed

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        # Register 'gamma' (risk-return trade-off parameter)
        # self.gamma = nn.Parameter(torch.FloatTensor(1).uniform_(0.037, 0.173))
        self.gamma = nn.Parameter(torch.FloatTensor(1).uniform_(0.02, 0.1))
        self.gamma.requires_grad = False

        # Record the model design: nominal, base or DRO
        if opt_layer == "nominal":
            self.model_type = "nom"
        elif opt_layer == "base_mod":
            self.model_type = "base_mod"
        else:
            # Register 'delta' (ambiguity sizing parameter) for DRO model
            if opt_layer == "hellinger":
                ub = (1 - 1 / (n_obs**0.5)) / 2
                lb = (1 - 1 / (n_obs**0.5)) / 10
            else:
                ub = (1 - 1 / n_obs) / 2
                lb = (1 - 1 / n_obs) / 10
            self.delta = nn.Parameter(torch.FloatTensor(1).uniform_(lb, ub))
            self.delta.requires_grad = False
            self.model_type = "dro"

        # LAYER: OLS linear prediction
        self.pred_layer = nn.Linear(n_x, n_y)
        self.pred_layer.weight.requires_grad = False
        self.pred_layer.bias.requires_grad = False

        # LAYER: Optimization
        # ``e2edro`` (and by extension ``cvxpylayers``) is only required when
        # constructing the optimization layer. Import here to avoid importing it
        # when using simpler models that do not rely on these packages.
        from . import e2edro as e2e

        self.opt_layer = e2e.OPT_LAYER_MAP[opt_layer](n_y, n_obs, e2e.RISK_FUNC_MAP[prisk])
        # self.opt_layer = e2e.nominal(n_y, n_obs, eval('rf.'+prisk))

    # -----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    # -----------------------------------------------------------------------------------------------
    def forward(self, X, Y):
        """Forward pass of the predict-then-optimize module

        The inputs 'X' are passed through the prediction layer to yield predictions 'Y_hat'. The
        residuals from prediction are then calcuclated as 'ep = Y - Y_hat'. Finally, the residuals
        are passed to the optimization layer to find the optimal decision z_star.

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data

        Other
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions

        Outputs
        y_hat: Prediction. (n_y x 1) vector of outputs of the prediction layer
        z_star: Optimal solution. (n_y x 1) vector of asset weights
        """
        batch = X.dim() == 3
        if not batch:
            X = X.unsqueeze(0)
            Y = Y.unsqueeze(0)

        Y_hat = self.pred_layer(X)

        ep = Y - Y_hat[:, :-1]
        y_hat = Y_hat[:, -1]

        # Optimization solver arguments (from CVXPY for SCS solver)
        solver_args = {"solve_method": "ECOS"}

        # Optimize z per scenario
        # Determine whether nominal or dro model
        z_list = []
        for i in range(X.size(0)):
            if self.model_type == "nom":
                (z,) = self.opt_layer(ep[i], y_hat[i], self.gamma, solver_args=solver_args)
            elif self.model_type == "dro":
                (z,) = self.opt_layer(
                    ep[i], y_hat[i], self.gamma, self.delta, solver_args=solver_args
                )
            elif self.model_type == "base_mod":
                (z,) = self.opt_layer(y_hat[i], solver_args=solver_args)
            z_list.append(z)
        z_star = torch.stack(z_list)

        if not batch:
            return z_star[0], y_hat[0]
        return z_star, y_hat

    # -----------------------------------------------------------------------------------------------
    # net_test: Test the e2e neural net
    # -----------------------------------------------------------------------------------------------
    def net_roll_test(self, X, Y, n_roll=4, batch_size: int = 1):
        """Neural net rolling window out-of-sample test

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data
        n_roll: Number of training periods (i.e., number of times to retrain the model)

        Output
        self.portfolio: add the backtest results to the e2e_net object
        """

        # Declare backtest object to hold the test results
        len_test = len(Y.test()) - Y.n_obs
        portfolio = pc.backtest(len_test, self.n_y, Y.test().index[Y.n_obs :])

        # Store initial train/test split
        init_split = Y.split

        # Window size
        win_size = init_split[1] / n_roll

        split = [0, 0]
        t = 0
        for i in range(n_roll):

            print(f"Out-of-sample window: {i+1} / {n_roll}")

            split[0] = init_split[0] + win_size * i
            if i < n_roll - 1:
                split[1] = win_size
            else:
                split[1] = 1 - split[0]

            X.split_update(split), Y.split_update(split)
            test_set = DataLoader(
                pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 0),
                batch_size=batch_size,
            )

            X_train, Y_train = X.train(), Y.train()
            X_train.insert(0, "ones", 1.0)

            X_train = torch.tensor(X_train.values, dtype=torch.double)
            Y_train = torch.tensor(Y_train.values, dtype=torch.double)

            Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
            Theta = Theta.T
            del X_train, Y_train

            with torch.no_grad():
                self.pred_layer.bias.copy_(Theta[:, 0])
                self.pred_layer.weight.copy_(Theta[:, 1:])

            # Test model
            with torch.no_grad():
                for j, (x, y, y_perf) in enumerate(test_set):

                    z_star, _ = self(x, y)

                    bsz = x.size(0)
                    portfolio.weights[t : t + bsz] = z_star.squeeze()
                    portfolio.rets[t : t + bsz] = (y_perf[:, 0, :] * portfolio.weights[t : t + bsz]).sum(dim=1)

                    t += bsz

        # Reset dataset
        X, Y = X.split_update(init_split), Y.split_update(init_split)

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        self.portfolio = portfolio


####################################################################################################
# Equal weight
####################################################################################################
class equal_weight:
    """Naive 'equally-weighted' portfolio construction module"""

    def __init__(self, n_x, n_y, n_obs):
        """Naive 'equally-weighted' portfolio construction module

        This object implements a basic equally-weighted investment strategy.

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        """
        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

    def state_dict(self) -> dict:
        """Return the state of the model.

        The equal-weight strategy has no learnable parameters, so this simply
        returns an empty dictionary. Implemented to provide a minimal interface
        compatible with ``torch.save``/``load`` utilities.
        """

        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state of the model.

        Because the strategy is stateless, this method does nothing but is
        present for API consistency.
        """

        return None

    # -----------------------------------------------------------------------------------------------
    # net_test: Test the e2e neural net
    # -----------------------------------------------------------------------------------------------
    def net_roll_test(self, X, Y, n_roll=4, batch_size: int = 1):
        """Neural net rolling window out-of-sample test

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data
        n_roll: Number of training periods (i.e., number of times to retrain the model)

        Output
        self.portfolio: add the backtest results to the e2e_net object
        """

        # Declare backtest object to hold the test results
        len_test = len(Y.test()) - Y.n_obs
        portfolio = pc.backtest(len_test, self.n_y, Y.test().index[Y.n_obs :])

        test_set = DataLoader(
            pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 0),
            batch_size=batch_size,
        )

        # Test model
        t = 0
        for j, (x, y, y_perf) in enumerate(test_set):

            bsz = x.size(0)
            portfolio.weights[t : t + bsz] = np.ones((bsz, self.n_y)) / self.n_y
            portfolio.rets[t : t + bsz] = (y_perf[:, 0, :] * portfolio.weights[t : t + bsz]).sum(dim=1)
            t += bsz

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        self.portfolio = portfolio


####################################################################################################
# Find gamma range
####################################################################################################
class gamma_range(nn.Module):
    """Simple way to approximately determine the appropriate values of gamma"""

    def __init__(self, n_x, n_y, n_obs):
        """Naive 'predict-then-optimize' portfolio construction module

        This NN module implements a linear prediction layer 'pred_layer' and an optimization layer
        'opt_layer'. The model is 'naive' since it optimizes each layer separately.

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: String. Portfolio risk function. Used in the opt_layer

        Output
        pred_then_opt: nn.Module object
        """
        super(gamma_range, self).__init__()

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        # LAYER: OLS linear prediction
        self.pred_layer = nn.Linear(n_x, n_y)
        self.pred_layer.weight.requires_grad = False
        self.pred_layer.bias.requires_grad = False

    # -----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    # -----------------------------------------------------------------------------------------------
    def forward(self, X, Y):
        """Forward pass of the predict-then-optimize module

        The inputs 'X' are passed through the prediction layer to yield predictions 'Y_hat'. The
        residuals from prediction are then calcuclated as 'ep = Y - Y_hat'. Finally, the residuals
        are passed to the optimization layer to find the optimal decision z_star.

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data

        Other
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions

        Outputs
        y_hat: Prediction. (n_y x 1) vector of outputs of the prediction layer
        z_star: Optimal solution. (n_y x 1) vector of asset weights
        """
        # Predict y_hat from x
        Y_hat = torch.stack([self.pred_layer(x_t) for x_t in X])

        # Calculate residuals and process them
        ep = Y - Y_hat[:-1]
        cov_ep = torch.cov(ep.T)

        # Find prediction
        y_hat = Y_hat[-1]

        # Set z=1/n per scenario
        z_star = torch.ones(self.n_y, dtype=torch.double) / self.n_y

        gamma = ((z_star.T @ cov_ep) @ z_star) / torch.abs(y_hat @ z_star)

        return gamma

    # -----------------------------------------------------------------------------------------------
    # gamma_eval: Find the range of gamma
    # -----------------------------------------------------------------------------------------------
    def gamma_eval(self, X, Y):
        """Use the equal weight portfolio and the nominal distribution to find appropriate
        values of gamma.

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data

        Output
        gamma: estimated gamma valules for each observation in the training set
        """

        # Initialize the prediction layer weights to OLS regression weights
        X_train, Y_train = X.train(), Y.train()
        X_train.insert(0, "ones", 1.0)

        X_train = torch.tensor(X_train.values, dtype=torch.double)
        Y_train = torch.tensor(Y_train.values, dtype=torch.double)

        Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
        Theta = Theta.T
        del X_train, Y_train

        with torch.no_grad():
            self.pred_layer.bias.copy_(Theta[:, 0])
            self.pred_layer.weight.copy_(Theta[:, 1:])

        # Construct training and validation DataLoader objects
        train_set = DataLoader(pc.SlidingWindow(X.train(), Y.train(), self.n_obs, 0))

        # Test model
        with torch.no_grad():
            gamma = []
            for t, (x, y, y_perf) in enumerate(train_set):
                gamma.append(self(x.squeeze(), y.squeeze()))

        return gamma
