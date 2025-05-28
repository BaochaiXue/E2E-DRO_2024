import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import Iterable
import yaml


def save_model(model: torch.nn.Module, path: str, cls: str, params: dict) -> None:
    """Save state dict and constructor parameters."""
    torch.save({"class": cls, "params": params, "state_dict": model.state_dict()}, path)


def load_model(path: str):
    """Load a model saved via ``save_model``."""
    data = torch.load(path, map_location="cpu")
    cls = data["class"]
    params = data["params"]
    if cls == "e2e_net":
        from e2edro import e2edro as e2e

        model = e2e.e2e_net(**params).double()
    elif cls == "pred_then_opt":
        from e2edro import BaseModels as bm

        model = bm.pred_then_opt(**params).double()
    elif cls == "equal_weight":
        from e2edro import BaseModels as bm

        model = bm.equal_weight(**params)
    else:
        raise ValueError(f"Unknown model class: {cls}")

    model.load_state_dict(data["state_dict"])
    return model

# Ensure the package is importable when running this script directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from e2edro import PlotFunctions as pf
from e2edro import BaseModels as bm
from e2edro import DataLoad as dl

cache_path = "./cache/exp/"
# Experiment 5 (with synthetic data)
####################################################################################################

# Path to cache the data, models and results
cache_path_exp5 = "./cache/exp5/"
# Ensure cache directories for experiment 5 also exist
os.makedirs(cache_path_exp5, exist_ok=True)
os.makedirs(os.path.join(cache_path_exp5, "plots"), exist_ok=True)

# Load hyperparameters from YAML config if available
_cfg_path = os.environ.get("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "config.yaml"))
if os.path.exists(_cfg_path):
    with open(_cfg_path, "r") as _f:
        _cfg = yaml.safe_load(_f) or {}
    HYP = _cfg.get("hyperparams", {})
else:
    HYP = {}

# ---------------------------------------------------------------------------------------------------
# Experiment 5: Load data
# ---------------------------------------------------------------------------------------------------

# Train, validation and test split percentage
split = HYP.get("split", [0.7, 0.3])

# Number of features and assets
n_x = HYP.get("n_x", 5)
n_y = HYP.get("n_y", 10)

# Number of observations per window and total number of observations
n_obs = HYP.get("n_obs", 100)
n_tot = HYP.get("n_tot", 1200)

# Synthetic data: randomly generate data from a linear model
X, Y = dl.synthetic_exp(n_x=n_x, n_y=n_y, n_obs=n_obs, n_tot=n_tot, split=split)

# ---------------------------------------------------------------------------------------------------
# Experiment 5: Initialize parameters
# ---------------------------------------------------------------------------------------------------

# Performance loss function and performance period 'v+1'
perf_loss = HYP.get("perf_loss", "sharpe_loss")
perf_period = HYP.get("perf_period", 13)

# Weight assigned to MSE prediction loss function
pred_loss_factor = HYP.get("pred_loss_factor", 0.5)

# Risk function (default set to variance)
prisk = HYP.get("prisk", "p_var")

# Robust decision layer to use: hellinger or tv
dr_layer = HYP.get("dr_layer", "hellinger")

# Determine whether to train the prediction weights Theta
train_pred = HYP.get("train_pred", True)

# List of learning rates to test
lr_list = HYP.get("lr_list", [0.005])

# List of total no. of epochs to test
epoch_list = HYP.get("epoch_list", [5])

# Load saved models (default is False)
use_cache = HYP.get("use_cache", False)

# ---------------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------------
if use_cache:
    nom_net_linear = load_model(cache_path_exp5 + "nom_net_linear.pt")
    nom_net_2layer = load_model(cache_path_exp5 + "nom_net_2layer.pt")
    nom_net_3layer = load_model(cache_path_exp5 + "nom_net_3layer.pt")
    dr_net_linear = load_model(cache_path_exp5 + "dr_net_linear.pt")
    dr_net_2layer = load_model(cache_path_exp5 + "dr_net_2layer.pt")
    dr_net_3layer = load_model(cache_path_exp5 + "dr_net_3layer.pt")
else:

    # Import heavy dependencies only when running the synthetic examples that rely on the differentiable optimization layers.
    from e2edro import e2edro as e2e

    # ***********************************************************************************************
    # Linear models
    # ***********************************************************************************************

    # For replicability, set the random seed for the numerical experiments
    set_seed = 2000

    # Nominal E2E linear
    nom_net_linear = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=train_pred,
        train_gamma=True,
        train_delta=True,
        set_seed=set_seed,
        opt_layer="nominal",
        perf_loss=perf_loss,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    nom_net_linear.net_cv(X, Y, lr_list, epoch_list, n_val=1)
    nom_net_linear.net_roll_test(X, Y, n_roll=1)
    save_model(
        nom_net_linear,
        cache_path + "nom_net_linear.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=train_pred,
            train_gamma=True,
            train_delta=True,
            set_seed=set_seed,
            opt_layer="nominal",
            perf_loss=perf_loss,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("nom_net_linear run complete")

    # DR E2E linear
    dr_net_linear = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=train_pred,
        train_gamma=True,
        train_delta=True,
        set_seed=set_seed,
        opt_layer=dr_layer,
        perf_loss=perf_loss,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    dr_net_linear.net_cv(X, Y, lr_list, epoch_list, n_val=1)
    dr_net_linear.net_roll_test(X, Y, n_roll=1)
    save_model(
        dr_net_linear,
        cache_path + "dr_net_linear.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=train_pred,
            train_gamma=True,
            train_delta=True,
            set_seed=set_seed,
            opt_layer=dr_layer,
            perf_loss=perf_loss,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("dr_net_linear run complete")

    # ***********************************************************************************************
    # 2-layer models
    # ***********************************************************************************************

    # For replicability, set the random seed for the numerical experiments
    set_seed = 3000

    # Nominal E2E 2-layer
    nom_net_2layer = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=train_pred,
        train_gamma=True,
        train_delta=True,
        pred_model="2layer",
        set_seed=set_seed,
        opt_layer="nominal",
        perf_loss=perf_loss,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    nom_net_2layer.net_cv(X, Y, lr_list, epoch_list, n_val=1)
    nom_net_2layer.net_roll_test(X, Y, n_roll=1)
    save_model(
        nom_net_2layer,
        cache_path + "nom_net_2layer.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=train_pred,
            train_gamma=True,
            train_delta=True,
            pred_model="2layer",
            set_seed=set_seed,
            opt_layer="nominal",
            perf_loss=perf_loss,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("nom_net_2layer run complete")

    # DR E2E 2-layer
    dr_net_2layer = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=train_pred,
        train_gamma=True,
        train_delta=True,
        pred_model="2layer",
        set_seed=set_seed,
        opt_layer=dr_layer,
        perf_loss=perf_loss,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    dr_net_2layer.net_cv(X, Y, lr_list, epoch_list, n_val=1)
    dr_net_2layer.net_roll_test(X, Y, n_roll=1)
    save_model(
        dr_net_2layer,
        cache_path + "dr_net_2layer.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=train_pred,
            train_gamma=True,
            train_delta=True,
            pred_model="2layer",
            set_seed=set_seed,
            opt_layer=dr_layer,
            perf_loss=perf_loss,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("dr_net_2layer run complete")

    # ***********************************************************************************************
    # 3-layer models
    # ***********************************************************************************************

    # For replicability, set the random seed for the numerical experiments
    set_seed = 4000

    # Nominal E2E 3-layer
    nom_net_3layer = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=train_pred,
        train_gamma=True,
        train_delta=True,
        pred_model="3layer",
        set_seed=set_seed,
        opt_layer="nominal",
        perf_loss=perf_loss,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    nom_net_3layer.net_cv(X, Y, lr_list, epoch_list, n_val=1)
    nom_net_3layer.net_roll_test(X, Y, n_roll=1)
    save_model(
        nom_net_3layer,
        cache_path + "nom_net_3layer.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=train_pred,
            train_gamma=True,
            train_delta=True,
            pred_model="3layer",
            set_seed=set_seed,
            opt_layer="nominal",
            perf_loss=perf_loss,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("nom_net_3layer run complete")

    # DR E2E 3-layer
    dr_net_3layer = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=train_pred,
        train_gamma=True,
        train_delta=True,
        pred_model="3layer",
        set_seed=set_seed,
        opt_layer=dr_layer,
        perf_loss=perf_loss,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    dr_net_3layer.net_cv(X, Y, lr_list, epoch_list, n_val=1)
    dr_net_3layer.net_roll_test(X, Y, n_roll=1)
    save_model(
        dr_net_3layer,
        cache_path + "dr_net_3layer.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=train_pred,
            train_gamma=True,
            train_delta=True,
            pred_model="3layer",
            set_seed=set_seed,
            opt_layer=dr_layer,
            perf_loss=perf_loss,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("dr_net_3layer run complete")

# ---------------------------------------------------------------------------------------------------
# Experiment 5: Results
# ---------------------------------------------------------------------------------------------------

# Validation results table
exp5_validation_table = pd.concat(
    (
        nom_net_linear.cv_results.round(4),
        dr_net_linear.cv_results.val_loss.round(4),
        nom_net_2layer.cv_results.val_loss.round(4),
        dr_net_2layer.cv_results.val_loss.round(4),
        nom_net_3layer.cv_results.val_loss.round(4),
        dr_net_3layer.cv_results.val_loss.round(4),
    ),
    axis=1,
)
exp5_validation_table.set_axis(
    [
        "eta",
        "Epochs",
        "Nom. (linear)",
        "DR (linear)",
        "Nom. (2-layer)",
        "DR (2-layer)",
        "Nom. (3-layer)",
        "DR (3-layer)",
    ],
    axis=1,
)

plt.rcParams["text.usetex"] = True
portfolio_names = [
    r"Nom. (linear)",
    r"DR (linear)",
    r"Nom. (2-layer)",
    r"DR (2-layer)",
    r"Nom. (3-layer)",
    r"DR (3-layer)",
]
portfolios = [
    nom_net_linear.portfolio,
    dr_net_linear.portfolio,
    nom_net_2layer.portfolio,
    dr_net_2layer.portfolio,
    nom_net_3layer.portfolio,
    dr_net_3layer.portfolio,
]

# Out-of-sample summary statistics table
exp5_fin_table = pf.fin_table(portfolios, portfolio_names)

# Wealth evolution plot
portfolio_colors = [
    "dodgerblue",
    "salmon",
    "dodgerblue",
    "salmon",
    "dodgerblue",
    "salmon",
]
pf.wealth_plot(
    portfolios,
    portfolio_names,
    portfolio_colors,
    nplots=3,
    path=cache_path + "plots/wealth_exp5.pdf",
)

# List of initial parameters
exp5_param_dict = dict(
    {
        "nom_net_linear": nom_net_linear.gamma_init,
        "dr_net_linear": [dr_net_linear.gamma_init, dr_net_linear.delta_init],
        "nom_net_2layer": nom_net_2layer.gamma_init,
        "dr_net_2layer": [dr_net_2layer.gamma_init, dr_net_2layer.delta_init],
        "nom_net_3layer": nom_net_3layer.gamma_init,
        "dr_net_3layer": [dr_net_3layer.gamma_init, dr_net_3layer.delta_init],
    }
)
