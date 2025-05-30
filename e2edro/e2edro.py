# E2E DRO Module
#
####################################################################################################
## Import libraries
####################################################################################################
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import RiskFunctions as rf
from . import LossFunctions as lf
from . import PortfolioClasses as pc
from . import DataLoad as dl

import psutil
from .OptimizationProblem import base_mod, nominal, tv, hellinger

num_cores = psutil.cpu_count()
torch.set_num_threads(num_cores)
if psutil.MACOS:
    num_cores = 0


####################################################################################################
# CvxpyLayers: Differentiable optimization layers (nominal and distributionally robust)
####################################################################################################
# Optimization layer definitions are imported from OptimizationProblem.py

# Mapping dictionaries for string based references
PERF_LOSS_MAP = {
    "single_period_loss": lf.single_period_loss,
    "single_period_over_var_loss": lf.single_period_over_var_loss,
    "sharpe_loss": lf.sharpe_loss,
}

RISK_FUNC_MAP = {
    "p_var": rf.p_var,
    "p_mad": rf.p_mad,
}

OPT_LAYER_MAP = {
    "base_mod": base_mod,
    "nominal": nominal,
    "tv": tv,
    "hellinger": hellinger,
}


####################################################################################################
# E2E neural network module
####################################################################################################
class e2e_net(nn.Module):
    """End-to-end DRO learning neural net module."""

    def __init__(
        self,
        n_x,
        n_y,
        n_obs,
        opt_layer="nominal",
        prisk="p_var",
        perf_loss="sharpe_loss",
        pred_model="linear",
        pred_loss_factor=0.5,
        perf_period=13,
        train_pred=True,
        train_gamma=True,
        train_delta=True,
        set_seed=None,
        cache_path="./cache/",
    ):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a DRO layer
        'opt_layer' based on a tractable convex formulation from Ben-Tal et al. (2013). 'delta' and
        'gamma' are declared as nn.Parameters so that they can be 'learned'.

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: String. Portfolio risk function. Used in the opt_layer
        opt_layer: String. Determines which CvxpyLayer-object to call for the optimization layer
        perf_loss: Performance loss function based on out-of-sample financial performance
        pred_loss_factor: Trade-off between prediction loss function and performance loss function.
            Set 'pred_loss_factor=None' to define the loss function purely as 'perf_loss'
        perf_period: Number of lookahead realizations used in 'perf_loss()'
        train_pred: Boolean. Choose if the prediction layer is learnable (or keep it fixed)
        train_gamma: Boolean. Choose if the risk appetite parameter gamma is learnable
        train_delta: Boolean. Choose if the robustness parameter delta is learnable
        set_seed: (Optional) Int. Set the random seed for replicability

        Output
        e2e_net: nn.Module object
        """
        super(e2e_net, self).__init__()

        # Set random seed (to be used for replicability of numerical experiments)
        if set_seed is not None:
            torch.manual_seed(set_seed)
            self.seed = set_seed

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        # Prediction loss function
        if pred_loss_factor is not None:
            self.pred_loss_factor = pred_loss_factor
            self.pred_loss = torch.nn.MSELoss()
        else:
            self.pred_loss = None

        # Define performance loss
        self.perf_loss = PERF_LOSS_MAP[perf_loss]

        # Number of time steps to evaluate the task loss
        self.perf_period = perf_period

        # Register 'gamma' (risk-return trade-off parameter)
        self.gamma = nn.Parameter(torch.FloatTensor(1).uniform_(0.02, 0.1))
        self.gamma.requires_grad = train_gamma
        self.gamma_init = self.gamma.item()

        # Record the model design: nominal, base or DRO
        if opt_layer == "nominal":
            self.model_type = "nom"
        elif opt_layer == "base_mod":
            self.gamma.requires_grad = False
            self.model_type = "base_mod"
        else:
            # Register 'delta' (ambiguity sizing parameter) for DR layer
            if opt_layer == "hellinger":
                ub = (1 - 1 / (n_obs**0.5)) / 2
                lb = (1 - 1 / (n_obs**0.5)) / 10
            else:
                ub = (1 - 1 / n_obs) / 2
                lb = (1 - 1 / n_obs) / 10
            self.delta = nn.Parameter(torch.FloatTensor(1).uniform_(lb, ub))
            self.delta.requires_grad = train_delta
            self.delta_init = self.delta.item()
            self.model_type = "dro"

        # LAYER: Prediction model
        self.pred_model = pred_model
        if pred_model == "linear":
            # Linear prediction model
            self.pred_layer = nn.Linear(n_x, n_y)
            self.pred_layer.weight.requires_grad = train_pred
            self.pred_layer.bias.requires_grad = train_pred
        elif pred_model == "2layer":
            # Neural net with 2 hidden layers
            self.pred_layer = nn.Sequential(
                nn.Linear(n_x, int(0.5 * (n_x + n_y))),
                nn.ReLU(),
                nn.Linear(int(0.5 * (n_x + n_y)), n_y),
                nn.ReLU(),
                nn.Linear(n_y, n_y),
            )
        elif pred_model == "3layer":
            # Neural net with 3 hidden layers
            self.pred_layer = nn.Sequential(
                nn.Linear(n_x, int(0.5 * (n_x + n_y))),
                nn.ReLU(),
                nn.Linear(int(0.5 * (n_x + n_y)), int(0.6 * (n_x + n_y))),
                nn.ReLU(),
                nn.Linear(int(0.6 * (n_x + n_y)), n_y),
                nn.ReLU(),
                nn.Linear(n_y, n_y),
            )

        # LAYER: Optimization model
        self.opt_layer_name = opt_layer
        self.prisk_name = prisk
        self.opt_layer = OPT_LAYER_MAP[opt_layer](n_y, n_obs, RISK_FUNC_MAP[prisk])

        # Store reference path to store model data
        self.cache_path = cache_path

        # Store initial model
        if train_gamma and train_delta:
            base_path = cache_path + self.model_type + "_initial_state_" + pred_model
        elif train_delta and not train_gamma:
            base_path = (
                cache_path
                + self.model_type
                + "_initial_state_"
                + pred_model
                + "_TrainGamma"
                + str(train_gamma)
            )
        elif train_gamma and not train_delta:
            base_path = (
                cache_path
                + self.model_type
                + "_initial_state_"
                + pred_model
                + "_TrainDelta"
                + str(train_delta)
            )
        else:
            base_path = (
                cache_path
                + self.model_type
                + "_initial_state_"
                + pred_model
                + "_TrainGamma"
                + str(train_gamma)
                + "_TrainDelta"
                + str(train_delta)
            )

        self.init_state_base = base_path
        self.init_state_path = base_path + ".pt"
        torch.save(self.state_dict(), self.init_state_path)

    # -----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    # -----------------------------------------------------------------------------------------------
    def forward(self, X, Y):
        """Forward pass of the NN module

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

        # Optimization solver arguments (from CVXPY for ECOS/SCS solver)
        solver_args = {"solve_method": "ECOS", "max_iters": 120, "abstol": 1e-7}
        # solver_args = {'solve_method': 'SCS', 'eps': 1e-7, 'acceleration_lookback': 5,
        # 'max_iters':20000}

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
    # net_train: Train the e2e neural net
    # -----------------------------------------------------------------------------------------------
    def net_train(self, train_set, val_set=None, epochs=None, lr=None):
        """Neural net training module

        Inputs
        train_set: SlidingWindow object containing features x, realizations y and performance
        realizations y_perf
        val_set: SlidingWindow object containing features x, realizations y and performance
        realizations y_perf
        epochs: Number of training epochs
        lr: learning rate

        Output
        Trained model
        (Optional) val_loss: Validation loss
        """

        # Assign number of epochs and learning rate
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr

        # Define the optimizer and its parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Number of elements in training set
        n_train = len(train_set.dataset)

        # Train the neural network
        for epoch in range(epochs):

            # TRAINING: forward + backward pass
            train_loss = 0
            optimizer.zero_grad()
            for t, (x, y, y_perf) in enumerate(train_set):

                # Forward pass: predict and optimize
                z_star, y_hat = self(x, y)

                # Loss function, y_perf is the performance realization
                bs = x.size(0)
                if self.pred_loss is None:
                    loss = self.perf_loss(z_star, y_perf)
                else:
                    loss = self.perf_loss(z_star, y_perf) + (
                        self.pred_loss_factor / self.n_y
                    ) * self.pred_loss(y_hat, y_perf[:, 0, :])
                loss = (bs / n_train) * loss

                # Backward pass: backpropagation
                loss.backward()

                # Accumulate loss of the fully trained model
                train_loss += loss.item()

            # Update parameters
            optimizer.step()

            # Ensure that gamma, delta > 0 after taking a descent step
            for name, param in self.named_parameters():
                if name == "gamma":
                    param.data.clamp_(0.0001)
                if name == "delta":
                    param.data.clamp_(0.0001)

        # Compute and return the validation loss of the model
        if val_set is not None:

            # Number of elements in validation set
            n_val = len(val_set.dataset)

            val_loss = 0
            with torch.no_grad():
                for t, (x, y, y_perf) in enumerate(val_set):

                    # Predict and optimize
                    z_val, y_val = self(x, y)

                    # Loss function
                    bs = x.size(0)
                    if self.pred_loss_factor is None:
                        loss = self.perf_loss(z_val, y_perf)
                    else:
                        loss = self.perf_loss(z_val, y_perf) + (
                            self.pred_loss_factor / self.n_y
                        ) * self.pred_loss(y_val, y_perf[:, 0, :])
                    loss = (bs / n_val) * loss

                    # Accumulate loss
                    val_loss += loss.item()

            return val_loss

    # -----------------------------------------------------------------------------------------------
    # net_cv: Cross validation of the e2e neural net for hyperparameter tuning
    # -----------------------------------------------------------------------------------------------
    def net_cv(self, X, Y, lr_list, epoch_list, n_val=4, batch_size: int = 1):
        """Neural net cross-validation module

        Inputs
        X: Features. TrainTest object of feature timeseries data
        Y: Realizations. TrainTest object of asset time series data
        epochs: number of training passes
        lr_list: List of candidate learning rates
        epoch_list: List of candidate number of epochs
        n_val: Number of validation folds from the training dataset

        Output
        Trained model
        """
        results = pc.CrossVal()
        X_temp = dl.TrainTest(X.train(), X.n_obs, [1, 0])
        Y_temp = dl.TrainTest(Y.train(), Y.n_obs, [1, 0])
        for epochs in epoch_list:
            for lr in lr_list:

                # Train the neural network
                print("================================================")
                print(f"Training E2E {self.model_type} model: lr={lr}, epochs={epochs}")

                val_loss_tot = []
                for i in range(n_val - 1, -1, -1):

                    # Partition training dataset into training and validation subset
                    split = [round(1 - 0.2 * (i + 1), 2), 0.2]
                    X_temp.split_update(split)
                    Y_temp.split_update(split)

                    # Construct training and validation DataLoader objects
                    train_set = DataLoader(
                        pc.SlidingWindow(
                            X_temp.train(), Y_temp.train(), self.n_obs, self.perf_period
                        ),
                        batch_size=batch_size,
                    )
                    val_set = DataLoader(
                        pc.SlidingWindow(
                            X_temp.test(), Y_temp.test(), self.n_obs, self.perf_period
                        ),
                        batch_size=batch_size,
                    )

                    # Reset learnable parameters gamma and delta
                    self.load_state_dict(
                        torch.load(self.init_state_path, weights_only=True)
                    )

                    if self.pred_model == "linear":
                        # Initialize the prediction layer weights to OLS regression weights
                        X_train, Y_train = X_temp.train(), Y_temp.train()
                        X_train.insert(0, "ones", 1.0)

                        X_train = torch.tensor(X_train.values, dtype=torch.double)
                        Y_train = torch.tensor(Y_train.values, dtype=torch.double)

                        Theta = torch.inverse(X_train.T @ X_train) @ (
                            X_train.T @ Y_train
                        )
                        Theta = Theta.T
                        del X_train, Y_train

                        with torch.no_grad():
                            self.pred_layer.bias.copy_(Theta[:, 0])
                            self.pred_layer.weight.copy_(Theta[:, 1:])

                    val_loss = self.net_train(
                        train_set, val_set=val_set, lr=lr, epochs=epochs
                    )
                    val_loss_tot.append(val_loss)

                    print(f"Fold: {n_val-i} / {n_val}, val_loss: {val_loss}")

                # Store results
                results.val_loss.append(np.mean(val_loss_tot))
                results.lr.append(lr)
                results.epochs.append(epochs)
                print("================================================")

        # Convert results to dataframe
        self.cv_results = results.df()
        self.cv_results.to_pickle(self.init_state_base + "_results.pkl")

        # Select and store the optimal hyperparameters
        idx = self.cv_results.val_loss.idxmin()
        self.lr = self.cv_results.lr[idx]
        self.epochs = self.cv_results.epochs[idx]

        # Print optimal parameters
        print(
            f"CV E2E {self.model_type} with hyperparameters: lr={self.lr}, epochs={self.epochs}"
        )

    # -----------------------------------------------------------------------------------------------
    # net_roll_test: Test the e2e neural net
    # -----------------------------------------------------------------------------------------------
    def net_roll_test(self, X, Y, n_roll=4, lr=None, epochs=None, batch_size: int = 1):
        """Neural net rolling window out-of-sample test

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data
        n_roll: Number of training periods (i.e., number of times to retrain the model)
        lr: Learning rate for test. If 'None', the optimal learning rate is loaded
        epochs: Number of epochs for test. If 'None', the optimal # of epochs is loaded

        Output
        self.portfolio: add the backtest results to the e2e_net object
        """

        # Declare backtest object to hold the test results
        len_test = len(Y.test()) - Y.n_obs
        portfolio = pc.backtest(len_test, self.n_y, Y.test().index[Y.n_obs :])

        # Store trained gamma and delta values
        if self.model_type == "nom":
            self.gamma_trained = []
        elif self.model_type == "dro":
            self.gamma_trained = []
            self.delta_trained = []

        # Store the squared L2-norm of the prediction weights and their difference from OLS weights
        if self.pred_model == "linear":
            self.theta_L2 = []
            self.theta_dist_L2 = []

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
            train_set = DataLoader(
                pc.SlidingWindow(X.train(), Y.train(), self.n_obs, self.perf_period),
                batch_size=batch_size,
            )
            test_set = DataLoader(
                pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 0),
                batch_size=batch_size,
            )

            # Reset learnable parameters gamma and delta
            self.load_state_dict(torch.load(self.init_state_path, weights_only=True))

            if self.pred_model == "linear":
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

            # Train model using all available data preceding the test window
            self.net_train(train_set, lr=lr, epochs=epochs)

            # Store trained values of gamma and delta
            if self.model_type == "nom":
                self.gamma_trained.append(self.gamma.item())
            elif self.model_type == "dro":
                self.gamma_trained.append(self.gamma.item())
                self.delta_trained.append(self.delta.item())

            # Store the squared L2 norm of theta and distance between theta and OLS weights
            if self.pred_model == "linear":
                theta_L2 = torch.sum(self.pred_layer.weight**2, axis=()) + torch.sum(
                    self.pred_layer.bias**2, axis=()
                )
                theta_dist_L2 = torch.sum(
                    (self.pred_layer.weight - Theta[:, 1:]) ** 2, axis=()
                ) + torch.sum((self.pred_layer.bias - Theta[:, 0]) ** 2, axis=())
                self.theta_L2.append(theta_L2)
                self.theta_dist_L2.append(theta_dist_L2)

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

    # -----------------------------------------------------------------------------------------------
    # load_cv_results: Load cross validation results
    # -----------------------------------------------------------------------------------------------
    def load_cv_results(self, cv_results):
        """Load cross validation results

        Inputs
        cv_results: pd.dataframe containing the cross validation results

        Outputs
        self.lr: Load the optimal learning rate
        self.epochs: Load the optimal number of epochs
        """

        # Store the cross validation results within the object
        self.cv_results = cv_results

        # Select and store the optimal hyperparameters
        idx = cv_results.val_loss.idxmin()
        self.lr = cv_results.lr[idx]
        self.epochs = cv_results.epochs[idx]

