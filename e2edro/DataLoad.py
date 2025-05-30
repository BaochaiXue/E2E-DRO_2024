# DataLoad module
#
####################################################################################################
## Import libraries
####################################################################################################
import torch
import torch.nn as nn
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import os

from .utils import robust_download


####################################################################################################
# TrainTest class
####################################################################################################
class TrainTest:
    def __init__(self, data: pd.DataFrame, n_obs: int, split: list[float]):
        """Object to hold the training, validation and testing datasets

        Inputs
        data: pandas dataframe with time series data
        n_obs: Number of observations per batch
        split: list of ratios that control the partition of data into training, testing and
        validation sets.

        Output. TrainTest object with fields and functions:
        data: Field. Holds the original pandas dataframe
        train(): Function. Returns a pandas dataframe with the training subset of observations
        """
        self.data = data
        self.n_obs = n_obs
        self.split = split

        n_obs_tot = self.data.shape[0]
        numel = n_obs_tot * np.cumsum(split)
        self.numel = [round(i) for i in numel]

    def split_update(self, split: list[float]) -> None:
        """Update the list outlining the split ratio of training, validation and testing"""
        self.split = split
        n_obs_tot = self.data.shape[0]
        numel = n_obs_tot * np.cumsum(split)
        self.numel = [round(i) for i in numel]

    def train(self) -> pd.DataFrame:
        """Return the training subset of observations"""
        return self.data[: self.numel[0]]

    def test(self) -> pd.DataFrame:
        """Return the test subset of observations"""
        return self.data[self.numel[0] - self.n_obs : self.numel[1]]


####################################################################################################
# Generate linear synthetic data
####################################################################################################
def synthetic(
    n_x: int = 5,
    n_y: int = 10,
    n_tot: int = 1200,
    n_obs: int = 104,
    split: list[float] | None = None,
    set_seed: int = 100,
) -> tuple[TrainTest, TrainTest]:
    """Generates synthetic (normally-distributed) asset and factor data

    Inputs
    n_x: Integer. Number of features
    n_y: Integer. Number of assets
    n_tot: Integer. Number of observations in the whole dataset
    n_obs: Integer. Number of observations per batch
    split: List of floats. Train-validation-test split as percentages (must sum up to one)
    set_seed: Integer. Used for replicability of the numpy RNG.

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """
    if split is None:
        split = [0.6, 0.4]

    np.random.seed(set_seed)

    # 'True' prediction bias and weights
    a = np.sort(np.random.rand(n_y) / 250) + 0.0001
    b = np.random.randn(n_x, n_y) / 5
    c = np.random.randn(int((n_x + 1) / 2), n_y)

    # Noise std dev
    s = np.sort(np.random.rand(n_y)) / 20 + 0.02

    # Syntehtic features
    X = np.random.randn(n_tot, n_x) / 50
    X2 = np.random.randn(n_tot, int((n_x + 1) / 2)) / 50

    # Synthetic outputs
    Y = a + X @ b + X2 @ c + s * np.random.randn(n_tot, n_y)

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)


####################################################################################################
# Generate non-linear synthetic data
####################################################################################################
def synthetic_nl(
    n_x: int = 5,
    n_y: int = 10,
    n_tot: int = 1200,
    n_obs: int = 104,
    split: list[float] | None = None,
    set_seed: int = 100,
) -> tuple[TrainTest, TrainTest]:
    """Generates synthetic (normally-distributed) factor data and mix them following a quadratic
    model of linear, squared and cross products to produce the asset data.

    Inputs
    n_x: Integer. Number of features
    n_y: Integer. Number of assets
    n_tot: Integer. Number of observations in the whole dataset
    n_obs: Integer. Number of observations per batch
    split: List of floats. Train-validation-test split as percentages (must sum up to one)
    set_seed: Integer. Used for replicability of the numpy RNG.

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """
    if split is None:
        split = [0.6, 0.4]

    np.random.seed(set_seed)

    # 'True' prediction bias and weights
    a = np.sort(np.random.rand(n_y) / 200) + 0.0005
    b = np.random.randn(n_x, n_y) / 4
    c = np.random.randn(int((n_x + 1) / 2), n_y)
    d = np.random.randn(n_x**2, n_y) / n_x

    # Noise std dev
    s = np.sort(np.random.rand(n_y)) / 20 + 0.02

    # Syntehtic features
    X = np.random.randn(n_tot, n_x) / 50
    X2 = np.random.randn(n_tot, int((n_x + 1) / 2)) / 50
    X_cross = 100 * (X[:, :, None] * X[:, None, :]).reshape(n_tot, n_x**2)
    X_cross = X_cross - X_cross.mean(axis=0)

    # Synthetic outputs
    Y = a + X @ b + X2 @ c + X_cross @ d + s * np.random.randn(n_tot, n_y)

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)


####################################################################################################
# Generate non-linear synthetic data
####################################################################################################
def synthetic_NN(
    n_x: int = 5,
    n_y: int = 10,
    n_tot: int = 1200,
    n_obs: int = 104,
    split: list[float] | None = None,
    set_seed: int = 45678,
) -> tuple[TrainTest, TrainTest]:
    """Generates synthetic (normally-distributed) factor data and mix them following a
    randomly-initialized 3-layer neural network.

    Inputs
    n_x: Integer. Number of features
    n_y: Integer. Number of assets
    n_tot: Integer. Number of observations in the whole dataset
    n_obs: Integer. Number of observations per batch
    split: List of floats. Train-validation-test split as percentages (must sum up to one)
    set_seed: Integer. Used for replicability of the numpy RNG.

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """
    if split is None:
        split = [0.6, 0.4]

    np.random.seed(set_seed)

    # Syntehtic features
    X = np.random.randn(n_tot, n_x) * 10 + 0.5

    # Initialize NN object
    synth = synthetic3layer(n_x, n_y, set_seed).double()

    # Synthetic outputs
    Y = synth(torch.from_numpy(X))

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y.detach().numpy()) / 10

    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)


####################################################################################################
# E2E neural network module
####################################################################################################
class synthetic3layer(nn.Module):
    """End-to-end DRO learning neural net module."""

    def __init__(self, n_x, n_y, set_seed):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a DRO layer
        'opt_layer' based on a tractable convex formulation from Ben-Tal et al. (2013). 'delta' and
        'gamma' are declared as nn.Parameters so that they can be 'learned'.

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model

        Output
        e2e_net: nn.Module object
        """
        super(synthetic3layer, self).__init__()

        # Set random seed (to be used for replicability of numerical experiments)
        torch.manual_seed(set_seed)

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

    # -----------------------------------------------------------------------------------------------
    # forward: forward pass of the synthetic3layer NN
    # -----------------------------------------------------------------------------------------------
    def forward(self, X):
        """Forward pass of the NN module

        Inputs
        X: Features. (n_obs x n_x) torch tensor with feature timeseries data

        Outputs
        Y: Syntheticly generated output. (n_obs x n_y) torch tensor of outputs
        """
        Y = torch.stack([self.pred_layer(x_t) for x_t in X])

        return Y


####################################################################################################
# Synthetic data with Gaussian and exponential noise terms
####################################################################################################
def synthetic_exp(
    n_x: int = 5,
    n_y: int = 10,
    n_tot: int = 1200,
    n_obs: int = 104,
    split: list[float] | None = None,
    set_seed: int = 123,
) -> tuple[TrainTest, TrainTest]:
    """Generate a synthetic dataset with Gaussian and exponential noise.

    Parameters
    ----------
    n_x : int, optional
        Number of features, by default 5.
    n_y : int, optional
        Number of assets, by default 10.
    n_tot : int, optional
        Total number of observations, by default 1200.
    n_obs : int, optional
        Window length for the sliding datasets, by default 104.
    split : list[float], optional
        Train-test split fractions, by default ``[0.6, 0.4]``.
    set_seed : int, optional
        Random seed for reproducibility, by default 123.

    Returns
    -------
    tuple[TrainTest, TrainTest]
        Pair of ``TrainTest`` objects for features ``X`` and targets ``Y``.
    """

    if split is None:
        split = [0.6, 0.4]

    np.random.seed(set_seed)

    # Exponential (shock) noise term
    exp_noise = (
        0.2
        * np.random.choice([-1, 0, 1], p=[0.15, 0.7, 0.15], size=(n_tot, n_y))
        * np.random.exponential(1, (n_tot, n_y))
    )
    exp_noise = exp_noise.clip(-0.3, 0.3)

    # Gaussian noise term
    gauss_noise = 0.2 * np.random.randn(n_tot, n_y)

    # 'True' prediction bias and weights
    alpha = np.sort(np.random.rand(n_y).clip(0.2, 1) / 1000)
    beta = np.random.randn(n_x, n_y).clip(-3, 3) / n_x

    # Syntehtic features
    X = np.random.randn(n_tot, n_x).clip(-3, 3) / 10

    # Synthetic outputs
    Y = (alpha + X @ beta + exp_noise + gauss_noise).clip(-0.2, 0.3) / 15

    # Convert to dataframes
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)


####################################################################################################
# Option 4: Factors from Kenneth French's data library and asset data from Yahoo Finance
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
# https://finance.yahoo.com
####################################################################################################
def AV(
    start: str,
    end: str,
    split: list,
    freq: str = "weekly",
    n_obs: int = 104,
    n_y=None,
    use_cache: bool = False,
    save_results: bool = False,
    AV_key: str = None,
):
    """Load data from Kenneth French's data library and from Yahoo Finance
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    https://www.alphavantage.co

    Parameters
    ----------
    start : str
        Start date of time series.
    end : str
        End date of time series.
    split : list
        Train-validation-test split as percentages .
    freq : str, optional
        Data frequency (daily, weekly, monthly). The default is 'weekly'.
    n_obs : int, optional
        Number of observations per batch. The default is 104.
    n_y : TYPE, optional
        Number of features to select. If None, the maximum number (8) is used. The default is None.
    use_cache : bool, optional
        State whether to load cached data or download data. The default is False.
    save_results : bool, optional
        State whether the data should be cached for future use. . The default is False.
    AV_key : str, optional
        Unused parameter kept for backwards compatibility. Asset prices are
        always downloaded using ``yfinance``.

    Returns
    -------
    X: TrainTest
        TrainTest object with feature data split into train, validation and test subsets.
    Y: TrainTest
        TrainTest object with asset data split into train, validation and test subsets.
    """

    cache_dir = os.path.join(".", "cache")
    if use_cache or save_results:
        os.makedirs(cache_dir, exist_ok=True)

    if use_cache:
        X = pd.read_pickle(os.path.join(cache_dir, f"factor_{freq}.pkl"))
        Y = pd.read_pickle(os.path.join(cache_dir, f"asset_{freq}.pkl"))
    else:
        tick_list = [
            "AAPL",
            "MSFT",
            "AMZN",
            "C",
            "JPM",
            "BAC",
            "XOM",
            "HAL",
            "MCD",
            "WMT",
            "COST",
            "CAT",
            "LMT",
            "JNJ",
            "PFE",
            "DIS",
            "VZ",
            "T",
            "ED",
            "NEM",
        ]

        if n_y is not None:
            tick_list = tick_list[:n_y]

        # Download asset data using yfinance with retry logic
        Y = robust_download(tick_list, start=start, end=end)
        Y = Y.pct_change().loc[start:end]

        # Download factor data
        ff_daily = pdr.get_data_famafrench(
            "F-F_Research_Data_Factors_daily", start=start, end=end
        )[0]
        rf_df = ff_daily.pop("RF") / 100
        mom_df = pdr.get_data_famafrench(
            "F-F_Momentum_Factor_daily", start=start, end=end
        )[0] / 100
        st_df = pdr.get_data_famafrench(
            "F-F_ST_Reversal_Factor_daily", start=start, end=end
        )[0] / 100
        lt_df = pdr.get_data_famafrench(
            "F-F_LT_Reversal_Factor_daily", start=start, end=end
        )[0] / 100

        # Concatenate factors as a pandas dataframe
        X = pd.concat([ff_daily / 100, mom_df, st_df, lt_df], axis=1)

        if freq == "weekly" or freq == "_weekly":
            # Convert daily returns to weekly returns
            Y = Y.resample("W-FRI").apply(lambda s: (s + 1).prod() - 1)
            X = X.resample("W-FRI").apply(lambda s: (s + 1).prod() - 1)

        if save_results:
            X.to_pickle(os.path.join(cache_dir, f"factor_{freq}.pkl"))
            Y.to_pickle(os.path.join(cache_dir, f"asset_{freq}.pkl"))

    if X.empty or Y.empty:
        raise ValueError(
            "Downloaded data is empty. Check your network connection or the provided date range."
        )

    if len(X) <= n_obs or len(Y) <= n_obs:
        raise ValueError(
            "Not enough data to create the requested sliding windows; try reducing 'n_obs' or expanding the date range."
        )

    # Partition dataset into training and testing sets. Lag the data by one observation
    return TrainTest(X[:-1], n_obs, split), TrainTest(Y[1:], n_obs, split)


####################################################################################################
# stats function
####################################################################################################
def statanalysis(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """Conduct a pairwise statistical significance analysis of each feature in X against each asset
    in Y.

    Parameters
    ----------
    X : pd.DataFrame
        Timeseries of features.
    Y : pd.DataFrame
        Timeseries of asset returns.

    Returns
    -------
    stats : pd.DataFrame
        Table of p-values obtained from regressing each individual feature against each individual
        asset.

    """

    stats = pd.DataFrame(columns=X.columns, index=Y.columns)
    for ticker in Y.columns:
        for feature in X.columns:
            # Align the feature and target by index and remove missing values to
            # avoid shape mismatches when constructing the design matrices.
            xy = pd.concat([X[feature], Y[ticker]], axis=1, join="inner").dropna()
            if xy.empty:
                stats.loc[ticker, feature] = np.nan
                continue

            stats.loc[ticker, feature] = (
                sm.OLS(xy[ticker].values, sm.add_constant(xy[feature]).values)
                .fit()
                .pvalues[1]
            )

    return stats.astype(float).round(2)
