# Distributionally Robust End-to-End Portfolio Construction
# Experiment 1 - General
####################################################################################################
# Import libraries
####################################################################################################
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import Iterable
import yaml


def save_model(model: torch.nn.Module, path: str, cls: str, params: dict) -> None:
    """Save only the state dict and constructor parameters."""
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


plt.close("all")

# Load hyperparameters from YAML config if available
_cfg_path = os.environ.get("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "config.yaml"))
if os.path.exists(_cfg_path):
    with open(_cfg_path, "r") as _f:
        _cfg = yaml.safe_load(_f) or {}
    HYP = _cfg.get("hyperparams", {})
else:
    HYP = {}

# Determine which experiments to run when executed via run_experiments.py
_exp_env = os.environ.get("EXP_LIST")
if _exp_env:
    EXP_LIST: Iterable[str] = [e.strip() for e in _exp_env.split(",") if e.strip()]
else:
    EXP_LIST = ["exp1", "exp2", "exp3", "exp4"]

# Make the code device-agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"

# Import E2E_DRO functions

# Path to cache the data, models and results
cache_path = "./cache/exp/"
# Ensure the cache directories exist so saving results does not fail
os.makedirs(cache_path, exist_ok=True)
os.makedirs(os.path.join(cache_path, "plots"), exist_ok=True)

####################################################################################################
# Experiments 1-4 (with hisotrical data): Load data
####################################################################################################

# Data frequency and start/end dates
freq = HYP.get("freq", "weekly")
start = HYP.get("start", "1998-01-01")
end = HYP.get("end", "2025-04-01")

# Train, validation and test split percentage
split = HYP.get("split", [0.6, 0.4])

# Number of observations per window
n_obs = HYP.get("n_obs", 104)

# Number of assets
n_y = HYP.get("n_y", 20)

# API key placeholder (not used).
# Asset prices are always downloaded via ``yfinance``.
AV_key = None

# Historical data: Download data (or load cached data)
X, Y = dl.AV(
    start,
    end,
    split,
    freq=freq,
    n_obs=n_obs,
    n_y=n_y,
    use_cache=False,
    save_results=False,
    AV_key=AV_key,
)

# Number of features and assets
n_x, n_y = X.data.shape[1], Y.data.shape[1]

# Statistical significance analysis of features vs targets
stats = dl.statanalysis(X.data, Y.data)

####################################################################################################
# E2E Learning System Run
####################################################################################################

# ---------------------------------------------------------------------------------------------------
# Initialize parameters
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

# List of learning rates to test
lr_list = HYP.get("lr_list", [0.005])

# List of total no. of epochs to test
epoch_list = HYP.get("epoch_list", [5])

# Batch size for DataLoader objects
batch_size = HYP.get("batch_size", 1)

# For replicability, set the random seed for the numerical experiments
set_seed = HYP.get("set_seed", 1000)

# Load saved models (default is False)
use_cache = HYP.get("use_cache", False)

# ---------------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------------

if use_cache:
    # Load cached models and backtest results
    ew_net = load_model(cache_path + "ew_net.pt")
    po_net = load_model(cache_path + "po_net.pt")
    base_net = load_model(cache_path + "base_net.pt")
    nom_net = load_model(cache_path + "nom_net.pt")
    dr_net = load_model(cache_path + "dr_net.pt")
    dr_po_net = load_model(cache_path + "dr_po_net.pt")
    dr_net_learn_delta = load_model(cache_path + "dr_net_learn_delta.pt")
    nom_net_learn_gamma = load_model(cache_path + "nom_net_learn_gamma.pt")
    dr_net_learn_gamma = load_model(cache_path + "dr_net_learn_gamma.pt")
    dr_net_learn_gamma_delta = load_model(cache_path + "dr_net_learn_gamma_delta.pt")
    nom_net_learn_theta = load_model(cache_path + "nom_net_learn_theta.pt")
    dr_net_learn_theta = load_model(cache_path + "dr_net_learn_theta.pt")

    base_net_ext = load_model(cache_path + "base_net_ext.pt")
    nom_net_ext = load_model(cache_path + "nom_net_ext.pt")
    dr_net_ext = load_model(cache_path + "dr_net_ext.pt")
    dr_net_learn_delta_ext = load_model(cache_path + "dr_net_learn_delta_ext.pt")
    nom_net_learn_gamma_ext = load_model(cache_path + "nom_net_learn_gamma_ext.pt")
    dr_net_learn_gamma_ext = load_model(cache_path + "dr_net_learn_gamma_ext.pt")
    nom_net_learn_theta_ext = load_model(cache_path + "nom_net_learn_theta_ext.pt")
    dr_net_learn_theta_ext = load_model(cache_path + "dr_net_learn_theta_ext.pt")

    dr_net_tv = load_model(cache_path + "dr_net_tv.pt")
    dr_net_tv_learn_delta = load_model(cache_path + "dr_net_tv_learn_delta.pt")
    dr_net_tv_learn_gamma = load_model(cache_path + "dr_net_tv_learn_gamma.pt")
    dr_net_tv_learn_theta = load_model(cache_path + "dr_net_tv_learn_theta.pt")
else:
    # Import here to avoid importing heavy dependencies when using only the
    # equal-weight example. ``e2edro`` requires ``cvxpylayers`` and ``diffcp``.
    from e2edro import e2edro as e2e

    # Exp 1: Equal weight portfolio
    ew_net = bm.equal_weight(n_x, n_y, n_obs)
    ew_net.net_roll_test(X, Y, n_roll=4, batch_size=batch_size)
    save_model(
        ew_net,
        cache_path + "ew_net.pt",
        "equal_weight",
        dict(n_x=n_x, n_y=n_y, n_obs=n_obs),
    )
    print("ew_net run complete")

    # Exp 1, 2, 3: Predict-then-optimize system
    po_net = bm.pred_then_opt(n_x, n_y, n_obs, set_seed=set_seed, prisk=prisk).double()
    po_net.net_roll_test(X, Y, batch_size=batch_size)
    save_model(
        po_net,
        cache_path + "po_net.pt",
        "pred_then_opt",
        dict(n_x=n_x, n_y=n_y, n_obs=n_obs, set_seed=set_seed, prisk=prisk),
    )
    print("po_net run complete")

    # Exp 1: Base E2E
    base_net = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=True,
        train_gamma=False,
        train_delta=False,
        set_seed=set_seed,
        opt_layer="base_mod",
        perf_loss=perf_loss,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    base_net.net_cv(X, Y, lr_list, epoch_list, batch_size=batch_size)
    base_net.net_roll_test(X, Y, batch_size=batch_size)
    save_model(
        base_net,
        cache_path + "base_net.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=True,
            train_gamma=False,
            train_delta=False,
            set_seed=set_seed,
            opt_layer="base_mod",
            perf_loss=perf_loss,
            cache_path=cache_path,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("base_net run complete")

    # Exp 1: Nominal E2E
    nom_net = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=True,
        train_gamma=True,
        train_delta=False,
        set_seed=set_seed,
        opt_layer="nominal",
        perf_loss=perf_loss,
        cache_path=cache_path,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    nom_net.net_cv(X, Y, lr_list, epoch_list, batch_size=batch_size)
    nom_net.net_roll_test(X, Y, batch_size=batch_size)
    save_model(
        nom_net,
        cache_path + "nom_net.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=True,
            train_gamma=True,
            train_delta=False,
            set_seed=set_seed,
            opt_layer="nominal",
            perf_loss=perf_loss,
            cache_path=cache_path,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("nom_net run complete")

    # Exp 1: DR E2E
    dr_net = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=True,
        train_gamma=True,
        train_delta=True,
        set_seed=set_seed,
        opt_layer=dr_layer,
        perf_loss=perf_loss,
        cache_path=cache_path,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    dr_net.net_cv(X, Y, lr_list, epoch_list, batch_size=batch_size)
    dr_net.net_roll_test(X, Y, batch_size=batch_size)
    save_model(
        dr_net,
        cache_path + "dr_net.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=True,
            train_gamma=True,
            train_delta=True,
            set_seed=set_seed,
            opt_layer=dr_layer,
            perf_loss=perf_loss,
            cache_path=cache_path,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("dr_net run complete")

    # Exp 2: DR predict-then-optimize system
    dr_po_net = bm.pred_then_opt(
        n_x, n_y, n_obs, set_seed=set_seed, prisk=prisk, opt_layer=dr_layer
    ).double()
    dr_po_net.net_roll_test(X, Y, batch_size=batch_size)
    save_model(
        dr_po_net,
        cache_path + "dr_po_net.pt",
        "pred_then_opt",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            set_seed=set_seed,
            prisk=prisk,
            opt_layer=dr_layer,
        ),
    )
    print("dr_po_net run complete")

    # Exp 2: DR E2E (fixed theta and gamma, learn delta)
    dr_net_learn_delta = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=False,
        train_gamma=False,
        train_delta=True,
        set_seed=set_seed,
        opt_layer=dr_layer,
        perf_loss=perf_loss,
        cache_path=cache_path,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    dr_net_learn_delta.net_cv(X, Y, lr_list, epoch_list, batch_size=batch_size)
    dr_net_learn_delta.net_roll_test(X, Y, batch_size=batch_size)
    save_model(
        dr_net_learn_delta,
        cache_path + "dr_net_learn_delta.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=False,
            train_gamma=False,
            train_delta=True,
            set_seed=set_seed,
            opt_layer=dr_layer,
            perf_loss=perf_loss,
            cache_path=cache_path,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("dr_net_learn_delta run complete")

    # Exp 3: Nominal E2E (fixed theta, learn gamma)
    nom_net_learn_gamma = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=False,
        train_gamma=True,
        train_delta=False,
        set_seed=set_seed,
        opt_layer="nominal",
        perf_loss=perf_loss,
        cache_path=cache_path,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    nom_net_learn_gamma.net_cv(X, Y, lr_list, epoch_list, batch_size=batch_size)
    nom_net_learn_gamma.net_roll_test(X, Y, batch_size=batch_size)
    save_model(
        nom_net_learn_gamma,
        cache_path + "nom_net_learn_gamma.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=False,
            train_gamma=True,
            train_delta=False,
            set_seed=set_seed,
            opt_layer="nominal",
            perf_loss=perf_loss,
            cache_path=cache_path,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("nom_net_learn_gamma run complete")

    # Exp 3: DR E2E (fixed theta, learn gamma, fixed delta)
    dr_net_learn_gamma = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=False,
        train_gamma=True,
        train_delta=False,
        set_seed=set_seed,
        opt_layer=dr_layer,
        perf_loss=perf_loss,
        cache_path=cache_path,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    dr_net_learn_gamma.net_cv(X, Y, lr_list, epoch_list, batch_size=batch_size)
    dr_net_learn_gamma.net_roll_test(X, Y, batch_size=batch_size)
    save_model(
        dr_net_learn_gamma,
        cache_path + "dr_net_learn_gamma.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=False,
            train_gamma=True,
            train_delta=False,
            set_seed=set_seed,
            opt_layer=dr_layer,
            perf_loss=perf_loss,
            cache_path=cache_path,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("dr_net_learn_gamma run complete")

    # Exp 4: Nominal E2E (learn theta, fixed gamma)
    nom_net_learn_theta = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=True,
        train_gamma=False,
        train_delta=False,
        set_seed=set_seed,
        opt_layer="nominal",
        perf_loss=perf_loss,
        cache_path=cache_path,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    nom_net_learn_theta.net_cv(X, Y, lr_list, epoch_list, batch_size=batch_size)
    nom_net_learn_theta.net_roll_test(X, Y, batch_size=batch_size)
    save_model(
        nom_net_learn_theta,
        cache_path + "nom_net_learn_theta.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=True,
            train_gamma=False,
            train_delta=False,
            set_seed=set_seed,
            opt_layer="nominal",
            perf_loss=perf_loss,
            cache_path=cache_path,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("nom_net_learn_theta run complete")

    # Exp 4: DR E2E (learn theta, fixed gamma and delta)
    dr_net_learn_theta = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=True,
        train_gamma=False,
        train_delta=False,
        set_seed=set_seed,
        opt_layer=dr_layer,
        perf_loss=perf_loss,
        cache_path=cache_path,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    dr_net_learn_theta.net_cv(X, Y, lr_list, epoch_list, batch_size=batch_size)
    dr_net_learn_theta.net_roll_test(X, Y, batch_size=batch_size)
    save_model(
        dr_net_learn_theta,
        cache_path + "dr_net_learn_theta.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=True,
            train_gamma=False,
            train_delta=False,
            set_seed=set_seed,
            opt_layer=dr_layer,
            perf_loss=perf_loss,
            cache_path=cache_path,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("dr_net_learn_theta run complete")

    # Exp 5: DR E2E (learn gamma, delta, fixed theta)
    dr_net_learn_gamma_delta = e2e.e2e_net(
        n_x,
        n_y,
        n_obs,
        prisk=prisk,
        train_pred=False,
        train_gamma=True,
        train_delta=True,
        set_seed=set_seed,
        opt_layer=dr_layer,
        perf_loss=perf_loss,
        cache_path=cache_path,
        perf_period=perf_period,
        pred_loss_factor=pred_loss_factor,
    ).double()
    dr_net_learn_gamma_delta.net_cv(X, Y, lr_list, epoch_list, batch_size=batch_size)
    dr_net_learn_gamma_delta.net_roll_test(X, Y, batch_size=batch_size)
    save_model(
        dr_net_learn_gamma_delta,
        cache_path + "dr_net_learn_gamma_delta.pt",
        "e2e_net",
        dict(
            n_x=n_x,
            n_y=n_y,
            n_obs=n_obs,
            prisk=prisk,
            train_pred=False,
            train_gamma=True,
            train_delta=True,
            set_seed=set_seed,
            opt_layer=dr_layer,
            perf_loss=perf_loss,
            cache_path=cache_path,
            perf_period=perf_period,
            pred_loss_factor=pred_loss_factor,
        ),
    )
    print("dr_net_learn_gamma_delta run complete")

####################################################################################################
# Merge objects with their extended-epoch counterparts,but we do have this part of the code to train
# the models for more epochs if needed.
####################################################################################################

# if use_cache:
#     portfolios = [
#         "base_net",
#         "nom_net",
#         "dr_net",
#         "dr_net_learn_delta",
#         "nom_net_learn_gamma",
#         "dr_net_learn_gamma",
#         "nom_net_learn_theta",
#         "dr_net_learn_theta",
#     ]

#     for portfolio in portfolios:
#         cv_combo = pd.concat(
#             [eval(portfolio).cv_results, eval(portfolio + "_ext").cv_results],
#             ignore_index=True,
#         )
#         eval(portfolio).load_cv_results(cv_combo)
#         if eval(portfolio).epochs > 50:
#             exec(portfolio + "=" + portfolio + "_ext")
#             eval(portfolio).load_cv_results(cv_combo)


####################################################################################################
# Numerical results
####################################################################################################

# ---------------------------------------------------------------------------------------------------
# Experiment 1: General
if "exp1" in EXP_LIST:
    # ---------------------------------------------------------------------------------------------------
    
    # Validation results table
    dr_net.cv_results = dr_net.cv_results.sort_values(
        ["epochs", "lr"], ascending=[True, True]
    ).reset_index(drop=True)
    exp1_validation_table = pd.concat(
        (
            base_net.cv_results.round(4),
            nom_net.cv_results.val_loss.round(4),
            dr_net.cv_results.val_loss.round(4),
        ),
        axis=1,
    )
    exp1_validation_table.set_axis(["eta", "Epochs", "Base", "Nom.", "DR"], axis=1)
    
    plt.rcParams["text.usetex"] = True
    portfolio_names = [r"EW", r"PO", r"Base", r"Nominal", r"DR"]
    portfolios = [
        ew_net.portfolio,
        po_net.portfolio,
        base_net.portfolio,
        nom_net.portfolio,
        dr_net.portfolio,
    ]
    
    # Out-of-sample summary statistics table
    exp1_fin_table = pf.fin_table(portfolios, portfolio_names)
    
    # Wealth evolution plot
    portfolio_colors = ["dimgray", "forestgreen", "goldenrod", "dodgerblue", "salmon"]
    pf.wealth_plot(
        portfolios,
        portfolio_names,
        portfolio_colors,
        path=cache_path + "plots/wealth_exp1.pdf",
    )
    pf.sr_bar(
        portfolios,
        portfolio_names,
        portfolio_colors,
        path=cache_path + "plots/sr_bar_exp1.pdf",
    )
    
    # List of initial parameters
    exp1_param_dict = dict(
        {
            "po_net": po_net.gamma.item(),
            "nom_net": nom_net.gamma_init,
            "dr_net": [dr_net.gamma_init, dr_net.delta_init],
        }
    )
    
    # Trained values for each out-of-sample investment period
    exp1_trained_vals = pd.DataFrame(
        zip(
            [nom_net.gamma_init] + nom_net.gamma_trained,
            [dr_net.gamma_init] + dr_net.gamma_trained,
            [dr_net.delta_init] + dr_net.delta_trained,
        ),
        columns=[r"Nom. $\gamma$", r"DR $\gamma$", r"DR $\delta$"],
    )
    
# ---------------------------------------------------------------------------------------------------
# Experiment 2: Learn delta
if "exp2" in EXP_LIST:
    # ---------------------------------------------------------------------------------------------------
    
    # Validation results table
    dr_net_learn_delta.cv_results = dr_net_learn_delta.cv_results.sort_values(
        ["epochs", "lr"], ascending=[True, True]
    ).reset_index(drop=True)
    exp2_validation_table = dr_net_learn_delta.cv_results.round(4)
    exp2_validation_table.set_axis(["eta", "Epochs", "DR (learn delta)"], axis=1)
    
    plt.rcParams["text.usetex"] = True
    portfolio_names = [r"PO", r"DR", r"DR (learn $\delta$)"]
    portfolios = [po_net.portfolio, dr_po_net.portfolio, dr_net_learn_delta.portfolio]
    
    # Out-of-sample summary statistics table
    exp2_fin_table = pf.fin_table(portfolios, portfolio_names)
    
    # Wealth evolution plots
    portfolio_colors = ["forestgreen", "dodgerblue", "salmon"]
    pf.wealth_plot(
        portfolios,
        portfolio_names,
        portfolio_colors,
        path=cache_path + "plots/wealth_exp2.pdf",
    )
    pf.sr_bar(
        portfolios,
        portfolio_names,
        portfolio_colors,
        path=cache_path + "plots/sr_bar_exp2.pdf",
    )
    
    # List of initial parameters
    exp2_param_dict = dict(
        {
            "po_net": po_net.gamma.item(),
            "dr_po_net": [dr_po_net.gamma.item(), dr_po_net.delta.item()],
            "dr_net_learn_delta": [
                dr_net_learn_delta.gamma_init,
                dr_net_learn_delta.delta_init,
            ],
        }
    )
    
    # Trained values for each out-of-sample investment period
    exp2_trained_vals = pd.DataFrame(
        [dr_net_learn_delta.delta_init] + dr_net_learn_delta.delta_trained,
        columns=["DR delta"],
    )
    
# ---------------------------------------------------------------------------------------------------
# Experiment 3: Learn gamma
if "exp3" in EXP_LIST:
    # ---------------------------------------------------------------------------------------------------
    
    # Validation results table
    dr_net_learn_gamma.cv_results = dr_net_learn_gamma.cv_results.sort_values(
        ["epochs", "lr"], ascending=[True, True]
    ).reset_index(drop=True)
    dr_net_learn_gamma_delta.cv_results = dr_net_learn_gamma_delta.cv_results.sort_values(
        ["epochs", "lr"], ascending=[True, True]
    ).reset_index(drop=True)
    exp3_validation_table = pd.concat(
        (
            nom_net_learn_gamma.cv_results.round(4),
            dr_net_learn_gamma.cv_results.val_loss.round(4),
            dr_net_learn_gamma_delta.cv_results.val_loss.round(4),
        ),
        axis=1,
    )
    exp3_validation_table.set_axis(
        [
            "eta",
            "Epochs",
            "Nom. (learn gamma)",
            "DR (learn gamma)",
            "DR (learn gamma + delta)",
        ],
        axis=1,
    )
    
    plt.rcParams["text.usetex"] = True
    portfolio_names = [r"PO", r"Nominal", r"DR"]
    portfolios = [
        po_net.portfolio,
        nom_net_learn_gamma.portfolio,
        dr_net_learn_gamma.portfolio,
    ]
    
    # Out-of-sample summary statistics table
    exp3_fin_table = pf.fin_table(portfolios, portfolio_names)
    
    # Wealth evolution plots
    portfolio_colors = ["forestgreen", "dodgerblue", "salmon"]
    pf.wealth_plot(
        portfolios,
        portfolio_names,
        portfolio_colors,
        path=cache_path + "plots/wealth_exp3.pdf",
    )
    pf.sr_bar(
        portfolios,
        portfolio_names,
        portfolio_colors,
        path=cache_path + "plots/sr_bar_exp3.pdf",
    )
    
    # List of initial parameters
    exp3_param_dict = dict(
        {
            "po_net": po_net.gamma.item(),
            "nom_net_learn_gamma": nom_net_learn_gamma.gamma_init,
            "dr_net_learn_gamma": [
                dr_net_learn_gamma.gamma_init,
                dr_net_learn_gamma.delta_init,
            ],
            "dr_net_learn_gamma_delta": [
                dr_net_learn_gamma_delta.gamma_init,
                dr_net_learn_gamma_delta.delta_init,
            ],
        }
    )
    
    # Trained values for each out-of-sample investment period
    exp3_trained_vals = pd.DataFrame(
        zip(
            [nom_net_learn_gamma.gamma_init] + nom_net_learn_gamma.gamma_trained,
            [dr_net_learn_gamma.gamma_init] + dr_net_learn_gamma.gamma_trained,
            [dr_net_learn_gamma_delta.gamma_init] + dr_net_learn_gamma_delta.gamma_trained,
            [dr_net_learn_gamma_delta.delta_init] + dr_net_learn_gamma_delta.delta_trained,
        ),
        columns=["Nom. gamma", "DR gamma", "DR gamma 2", "DR delta"],
    )
    
    # ---------------------------------------------------------------------------------------------------
# Experiment 4: Learn theta
if "exp4" in EXP_LIST:
    # ---------------------------------------------------------------------------------------------------
    
    # Validation results table
    dr_net_learn_theta.cv_results = dr_net_learn_theta.cv_results.sort_values(
        ["epochs", "lr"], ascending=[True, True]
    ).reset_index(drop=True)
    exp4_validation_table = pd.concat(
        (
            base_net.cv_results.round(4),
            nom_net_learn_theta.cv_results.val_loss.round(4),
            dr_net_learn_theta.cv_results.val_loss.round(4),
        ),
        axis=1,
    )
    exp4_validation_table.set_axis(["eta", "Epochs", "Base", "Nom.", "DR"], axis=1)
    
    plt.rcParams["text.usetex"] = True
    portfolio_names = [r"PO", r"Base", r"Nominal", r"DR"]
    portfolios = [
        po_net.portfolio,
        base_net.portfolio,
        nom_net_learn_theta.portfolio,
        dr_net_learn_theta.portfolio,
    ]
    
    # Out-of-sample summary statistics table
    exp4_fin_table = pf.fin_table(portfolios, portfolio_names)
    
    # Wealth evolution plots
    portfolio_colors = ["forestgreen", "goldenrod", "dodgerblue", "salmon"]
    pf.wealth_plot(
        portfolios,
        portfolio_names,
        portfolio_colors,
        path=cache_path + "plots/wealth_exp4.pdf",
    )
    pf.sr_bar(
        portfolios,
        portfolio_names,
        portfolio_colors,
        path=cache_path + "plots/sr_bar_exp4.pdf",
    )
    
    # List of initial parameters
    exp4_param_dict = dict(
        {
            "po_net": po_net.gamma.item(),
            "nom_net_learn_theta": nom_net_learn_theta.gamma_init,
            "dr_net_learn_theta": [
                dr_net_learn_theta.gamma_init,
                dr_net_learn_theta.delta_init,
            ],
        }
    )
    
    # Trained values for each out-of-sample investment period
    exp4_trained_vals = pd.DataFrame(
        zip(
            nom_net_learn_theta.gamma_trained,
            dr_net_learn_theta.gamma_trained,
            dr_net_learn_theta.delta_trained,
        ),
        columns=["Nom. gamma", "DR gamma", "DR delta"],
    )
    
    # ---------------------------------------------------------------------------------------------------
# Aggregate Validation Results
# ---------------------------------------------------------------------------------------------------

validation_table = pd.concat(
    (
        base_net.cv_results.round(4),
        nom_net.cv_results.val_loss.round(4),
        nom_net_learn_gamma.cv_results.val_loss.round(4),
        nom_net_learn_theta.cv_results.val_loss.round(4),
        dr_net.cv_results.val_loss.round(4),
        dr_net_learn_delta.cv_results.val_loss.round(4),
        dr_net_learn_gamma.cv_results.val_loss.round(4),
        dr_net_learn_gamma_delta.cv_results.val_loss.round(4),
        dr_net_learn_theta.cv_results.val_loss.round(4),
    ),
    axis=1,
)
validation_table.set_axis(
    [
        "eta",
        "Epochs",
        "Base",
        "Nom.",
        "Nom. (gamma)",
        "Nom. (theta)",
        "DR",
        "DR (delta)",
        "DR (gamma)",
        "DR (gamma+delta)",
        "DR (theta)",
    ],
    axis=1,
)

####################################################################################################
