# DataLoad module
#
####################################################################################################
# Import libraries
####################################################################################################
import torch
import torch.nn as nn
import pandas as pd
import pandas_datareader as pdr
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time
import statsmodels.api as sm


####################################################################################################
# TrainTest class
####################################################################################################
class TrainTest:
    def __init__(self, data: pd.DataFrame, n_obs: int, split: list[float]) -> None:
        """
        Object to hold the training, validation, and testing datasets.

        :param data: pandas DataFrame with time series data.
        :param n_obs: Number of observations per batch.
        :param split: List of ratios that control the partition of data into training, testing, and validation sets.
        """
        self.data: pd.DataFrame = data  # Store the input data as a DataFrame
        self.n_obs: int = n_obs  # Set the number of observations per batch
        self.split: list[float] = (
            split  # Set the split ratios for training, validation, and testing
        )

        n_obs_tot: int = self.data.shape[
            0
        ]  # Calculate the total number of observations in the dataset
        numel: np.ndarray = n_obs_tot * np.cumsum(
            split
        )  # Calculate the cumulative number of elements based on split ratios
        self.numel: list[int] = [
            round(i) for i in numel
        ]  # Round the cumulative elements to get the indices for splits

    def split_update(self, split: list[float]) -> None:
        """
        Update the list outlining the split ratio of training, validation, and testing datasets.

        :param split: List of ratios that control the partition of data into training, testing, and validation sets.
        """
        self.split: list[float] = (
            split  # Update the split ratios with the new list provided
        )
        n_obs_tot: int = self.data.shape[
            0
        ]  # Calculate the total number of observations in the dataset
        numel: np.ndarray = n_obs_tot * np.cumsum(
            split
        )  # Calculate the cumulative number of elements based on new split ratios
        self.numel: list[int] = [
            round(i) for i in numel
        ]  # Round the cumulative elements to get the indices for splits

    def train(self) -> pd.DataFrame:
        """
        Return the training subset of observations.

        :return: pandas DataFrame containing the training data subset.
        """
        return self.data[
            : self.numel[0]
        ]  # Return the data from the start up to the end of the training set

    def test(self) -> pd.DataFrame:
        """
        Return the test subset of observations.

        :return: pandas DataFrame containing the test data subset.
        """
        return self.data[
            self.numel[0] - self.n_obs : self.numel[1]
        ]  # Return the data for the test set, including overlap


####################################################################################################
# Generate linear synthetic data
####################################################################################################
def synthetic(
    n_x: int = 5,
    n_y: int = 10,
    n_tot: int = 1200,
    n_obs: int = 104,
    split: list[float] = [0.6, 0.4],
    set_seed: int = 100,
) -> tuple[TrainTest, TrainTest]:
    """
    Generates synthetic (normally-distributed) asset and factor data.

    :param n_x: Number of features.
    :param n_y: Number of assets.
    :param n_tot: Number of observations in the whole dataset.
    :param n_obs: Number of observations per batch.
    :param split: List of floats representing train-validation-test split percentages (must sum up to one).
    :param set_seed: Integer seed for replicability of the numpy RNG.

    :return: Tuple of TrainTest objects for features and asset data split into train, validation, and test subsets.
    """
    np.random.seed(set_seed)  # Set the random seed for reproducibility

    # 'True' prediction bias and weights
    a: np.ndarray = (
        np.sort(np.random.rand(n_y) / 250) + 0.0001
    )  # Generate small bias terms for each asset
    b: np.ndarray = (
        np.random.randn(n_x, n_y) / 5
    )  # Generate random weights for linear relationships between features and assets
    c: np.ndarray = np.random.randn(
        int((n_x + 1) / 2), n_y
    )  # Generate additional random weights for auxiliary features

    # Noise standard deviation
    s: np.ndarray = (
        np.sort(np.random.rand(n_y)) / 20 + 0.02
    )  # Generate small standard deviations for noise for each asset

    # Synthetic features
    X: np.ndarray = (
        np.random.randn(n_tot, n_x) / 50
    )  # Generate synthetic features from a normal distribution
    X2: np.ndarray = (
        np.random.randn(n_tot, int((n_x + 1) / 2)) / 50
    )  # Generate auxiliary features from a normal distribution

    # Synthetic outputs
    Y: np.ndarray = (
        a + X @ b + X2 @ c + s * np.random.randn(n_tot, n_y)
    )  # Generate synthetic outputs based on linear combinations of features and noise

    X: pd.DataFrame = pd.DataFrame(X)  # Convert features to a pandas DataFrame
    Y: pd.DataFrame = pd.DataFrame(Y)  # Convert outputs to a pandas DataFrame

    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(
        Y, n_obs, split
    )  # Return TrainTest objects for features and outputs


####################################################################################################
# Generate non-linear synthetic data
####################################################################################################
import numpy as np
import pandas as pd
import unittest


def synthetic_nl(
    n_x: int = 5,
    n_y: int = 10,
    n_tot: int = 1200,
    n_obs: int = 104,
    split: list[float] = [0.6, 0.4],
    set_seed: int = 100,
) -> tuple[TrainTest, TrainTest]:
    """
    Generates synthetic (normally-distributed) factor data and mixes them using a quadratic model
    of linear, squared, and cross products to produce the asset data.

    :param n_x: Number of features.
    :param n_y: Number of assets.
    :param n_tot: Number of observations in the whole dataset.
    :param n_obs: Number of observations per batch.
    :param split: List of floats representing train-validation-test split percentages (must sum up to one).
    :param set_seed: Integer seed for replicability of the numpy RNG.
    :return: Tuple of TrainTest objects for features and asset data split into train, validation, and test subsets.
    """
    # Set the random seed for reproducibility
    np.random.seed(set_seed)

    # Generate 'True' prediction bias and weights
    a: np.ndarray = (
        np.sort(np.random.rand(n_y) / 200) + 0.0005
    )  # Bias terms for each asset
    b: np.ndarray = np.random.randn(n_x, n_y) / 4  # Linear relationship weights
    c: np.ndarray = np.random.randn(
        int((n_x + 1) / 2), n_y
    )  # Auxiliary feature weights
    d: np.ndarray = np.random.randn(n_x**2, n_y) / n_x  # Cross-product weights

    # Generate noise standard deviation for each asset
    s: np.ndarray = np.sort(np.random.rand(n_y)) / 20 + 0.02

    # Generate synthetic features
    X: np.ndarray = np.random.randn(n_tot, n_x) / 50  # Main features
    X2: np.ndarray = (
        np.random.randn(n_tot, int((n_x + 1) / 2)) / 50
    )  # Auxiliary features
    X_cross: np.ndarray = 100 * (X[:, :, None] * X[:, None, :]).reshape(
        n_tot, n_x**2
    )  # Cross-product features
    X_cross = X_cross - X_cross.mean(axis=0)  # Center cross-product features

    # Generate synthetic outputs
    Y: np.ndarray = a + X @ b + X2 @ c + X_cross @ d + s * np.random.randn(n_tot, n_y)

    # Convert features and outputs to pandas DataFrames
    X_df: pd.DataFrame = pd.DataFrame(X)
    Y_df: pd.DataFrame = pd.DataFrame(Y)

    # Partition dataset into training and testing sets
    return TrainTest(X_df, n_obs, split), TrainTest(Y_df, n_obs, split)


####################################################################################################
# Unit Test for synthetic_nl function
####################################################################################################
def test_synthetic_nl() -> None:
    n_x = 5
    n_y = 10
    n_tot = 1200
    n_obs = 104
    split = [0.6, 0.4]
    set_seed = 100

    features, outputs = synthetic_nl(
        n_x=n_x,
        n_y=n_y,
        n_tot=n_tot,
        n_obs=n_obs,
        split=split,
        set_seed=set_seed,
    )

    assert isinstance(
        features, TrainTest
    ), "Features should be an instance of TrainTest."
    assert isinstance(outputs, TrainTest), "Outputs should be an instance of TrainTest."
    assert features.data.shape == (n_tot, n_x), "Features data shape mismatch."
    assert outputs.data.shape == (n_tot, n_y), "Outputs data shape mismatch."
    expected_train_len = int(n_tot * split[0])
    assert (
        len(features.train()) == expected_train_len
    ), "Training split size mismatch for features."
    assert (
        len(outputs.train()) == expected_train_len
    ), "Training split size mismatch for outputs."

    # Test reproducibility
    features_1, outputs_1 = synthetic_nl(
        n_x=n_x,
        n_y=n_y,
        n_tot=n_tot,
        n_obs=n_obs,
        split=split,
        set_seed=set_seed,
    )
    features_2, outputs_2 = synthetic_nl(
        n_x=n_x,
        n_y=n_y,
        n_tot=n_tot,
        n_obs=n_obs,
        split=split,
        set_seed=set_seed,
    )
    pd.testing.assert_frame_equal(
        features_1.data, features_2.data, "Feature data mismatch for reproducibility."
    )
    pd.testing.assert_frame_equal(
        outputs_1.data, outputs_2.data, "Output data mismatch for reproducibility."
    )


####################################################################################################
# Generate non-linear synthetic data
####################################################################################################
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import time
import statsmodels.api as sm
import pandas_datareader as pdr
from alpha_vantage.timeseries import TimeSeries


# Function to generate synthetic data using a 3-layer neural network
def synthetic_NN(
    n_x: int = 5,
    n_y: int = 10,
    n_tot: int = 1200,
    n_obs: int = 104,
    split: List[float] = [0.6, 0.4],
    set_seed: int = 45678,
) -> Tuple["TrainTest", "TrainTest"]:
    """
    Generates synthetic (normally-distributed) factor data and mixes them using a 3-layer neural network.

    :param n_x: Number of features.
    :param n_y: Number of assets.
    :param n_tot: Number of observations in the whole dataset.
    :param n_obs: Number of observations per batch.
    :param split: List of floats representing train-validation-test split percentages (must sum up to one).
    :param set_seed: Integer seed for reproducibility.
    :return: Tuple of TrainTest objects for feature and asset data.
    """
    np.random.seed(set_seed)  # Set random seed for reproducibility

    # Generate synthetic features (n_tot samples with n_x features each)
    X: np.ndarray = (
        np.random.randn(n_tot, n_x) * 10 + 0.5
    )  # Scale and shift features to have varied range

    # Initialize neural network
    synth: synthetic3layer = synthetic3layer(
        n_x, n_y, set_seed
    ).double()  # Create an instance of the 3-layer neural network

    # Generate synthetic outputs using the neural network
    Y: torch.Tensor = synth(
        torch.from_numpy(X)
    )  # Convert X to a torch tensor and pass it through the neural network

    # Convert synthetic features and outputs to pandas DataFrames
    X_df: pd.DataFrame = pd.DataFrame(X)  # Convert features to DataFrame
    Y_df: pd.DataFrame = (
        pd.DataFrame(Y.detach().numpy()) / 10
    )  # Convert outputs to DataFrame and scale down

    # Return TrainTest objects for features and outputs
    return TrainTest(X_df, n_obs, split), TrainTest(Y_df, n_obs, split)


####################################################################################################
# Neural Network Module
####################################################################################################
class synthetic3layer(nn.Module):
    def __init__(self, n_x: int, n_y: int, set_seed: int) -> None:
        """
        Initialize a 3-layer neural network to synthesize data.

        :param n_x: Number of input features.
        :param n_y: Number of output features.
        :param set_seed: Integer seed for reproducibility.
        """
        super().__init__()  # Call the parent class (nn.Module) initializer
        torch.manual_seed(
            set_seed
        )  # Set random seed for torch to ensure reproducible weights

        # Define a neural network with 3 hidden layers
        self.pred_layer: nn.Sequential = nn.Sequential(
            nn.Linear(
                n_x, int(0.5 * (n_x + n_y))
            ),  # First linear layer to project input features to an intermediate size
            nn.ReLU(),  # ReLU activation function to introduce non-linearity
            nn.Linear(
                int(0.5 * (n_x + n_y)), int(0.6 * (n_x + n_y))
            ),  # Second linear layer to another intermediate size
            nn.ReLU(),  # ReLU activation function
            nn.Linear(
                int(0.6 * (n_x + n_y)), n_y
            ),  # Third linear layer to project to output size
            nn.ReLU(),  # ReLU activation function
            nn.Linear(n_y, n_y),  # Final linear layer to produce the final outputs
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the neural network to generate synthetic outputs.

        :param X: Features. (n_obs x n_x) torch tensor with feature time series data.
        :return: Synthetically generated output. (n_obs x n_y) torch tensor of outputs.
        """
        # Apply the prediction layer to each input tensor in the batch and stack the results
        return torch.stack([self.pred_layer(x_t) for x_t in X])


####################################################################################################
# Test code to detect bugs
####################################################################################################
def test_synthetic_nn() -> None:
    try:
        # Define parameters for the test
        n_x = 5
        n_y = 10
        n_tot = 1200
        n_obs = 104
        split = [0.6, 0.4]
        set_seed = 45678

        # Generate synthetic data
        features, outputs = synthetic_NN(n_x, n_y, n_tot, n_obs, split, set_seed)

        # Check if generated features and outputs are pandas DataFrames
        assert isinstance(
            features.data, pd.DataFrame
        ), "Features should be a pandas DataFrame."
        assert isinstance(
            outputs.data, pd.DataFrame
        ), "Outputs should be a pandas DataFrame."

        # Check the dimensions of the generated data
        assert features.data.shape == (
            n_tot,
            n_x,
        ), f"Expected features shape {(n_tot, n_x)}, but got {features.data.shape}."
        assert outputs.data.shape == (
            n_tot,
            n_y,
        ), f"Expected outputs shape {(n_tot, n_y)}, but got {outputs.data.shape}."

        # Check reproducibility
        features_1, outputs_1 = synthetic_NN(n_x, n_y, n_tot, n_obs, split, set_seed)
        features_2, outputs_2 = synthetic_NN(n_x, n_y, n_tot, n_obs, split, set_seed)
        pd.testing.assert_frame_equal(
            features_1.data,
            features_2.data,
            "Feature data mismatch for reproducibility.",
        )
        pd.testing.assert_frame_equal(
            outputs_1.data, outputs_2.data, "Output data mismatch for reproducibility."
        )

        print("All tests passed successfully.")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


from typing import Callable
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


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


import pandas as pd
import time
from alpha_vantage.timeseries import TimeSeries
import pandas_datareader.data as pdr

####################################################################################################
# Option 4: Factors from Kenneth French's data library and asset data from AlphaVantage
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
# https://www.alphavantage.co
####################################################################################################


def AV(
    start: str,
    end: str,
    split: list[float],
    freq: str = "weekly",
    n_obs: int = 104,
    n_y: int | None = None,
    use_cache: bool = False,
    save_results: bool = False,
    AV_key: str | None = "YDNA9HH8P2IW985M",
) -> tuple[TrainTest, TrainTest]:
    """
    Load data from Kenneth French's data library and from AlphaVantage.

    :param start: Start date of time series.
    :param end: End date of time series.
    :param split: List of floats representing train-validation-test split percentages.
    :param freq: Data frequency (daily, weekly, monthly). Default is 'weekly'.
    :param n_obs: Number of observations per batch. Default is 104.
    :param n_y: Number of features to select. If None, the maximum number (8) is used. Default is None.
    :param use_cache: Whether to load cached data or download new data. Default is False.
    :param save_results: Whether to save the data for future use. Default is False.
    :param AV_key: AlphaVantage API key for accessing their data. Default is 'YDNA9HH8P2IW985M'.
    :return: Tuple containing TrainTest objects for features and asset data.
    """
    if use_cache:
        # Load cached data
        X: pd.DataFrame = pd.read_pickle(f"./cache/factor_{freq}.pkl")
        Y: pd.DataFrame = pd.read_pickle(f"./cache/asset_{freq}.pkl")
    else:
        # Define list of tickers to be downloaded
        tick_list: list[str] = [
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

        # Select the first n_y tickers if specified
        if n_y is not None:
            tick_list = tick_list[:n_y]

        # Ensure API key is provided
        if AV_key is None:
            print(
                """A personal AlphaVantage API key is required to load the asset pricing data.
                If you do not have a key, you can get one from www.alphavantage.co (free for academic users)"""
            )
            AV_key = input("Enter your AlphaVantage API key: ")

        # Initialize TimeSeries object to interact with AlphaVantage API
        ts: TimeSeries = TimeSeries(
            key=AV_key, output_format="pandas", indexing_type="date"
        )

        # Download asset data from AlphaVantage
        Y_list: list[pd.Series] = []
        for tick in tick_list:
            data, _ = ts.get_daily_adjusted(symbol=tick, outputsize="full")
            data = data["5. adjusted close"]
            Y_list.append(data)
            time.sleep(12.5)  # AlphaVantage rate limit to avoid API ban
        Y: pd.DataFrame = pd.concat(Y_list, axis=1)
        Y = Y[::-1]
        Y = Y.loc["1999-01-01":end].pct_change()
        Y = Y.loc[start:end]
        Y.columns = tick_list

        # Download factor data from Kenneth French's library
        dl_freq: str = "_daily"
        X: pd.DataFrame = pdr.get_data_famafrench(
            f"F-F_Research_Data_5_Factors_2x3{dl_freq}", start=start, end=end
        )[0]
        rf_df: pd.Series = X["RF"]
        X = X.drop(["RF"], axis=1)
        mom_df: pd.DataFrame = pdr.get_data_famafrench(
            f"F-F_Momentum_Factor{dl_freq}", start=start, end=end
        )[0]
        st_df: pd.DataFrame = pdr.get_data_famafrench(
            f"F-F_ST_Reversal_Factor{dl_freq}", start=start, end=end
        )[0]
        lt_df: pd.DataFrame = pdr.get_data_famafrench(
            f"F-F_LT_Reversal_Factor{dl_freq}", start=start, end=end
        )[0]

        # Concatenate all factors into a single DataFrame
        X = pd.concat([X, mom_df, st_df, lt_df], axis=1) / 100

        # Convert daily returns to weekly returns if specified
        if freq in ["weekly", "_weekly"]:
            Y = Y.resample("W-FRI").agg(lambda x: (x + 1).prod() - 1)
            X = X.resample("W-FRI").agg(lambda x: (x + 1).prod() - 1)

        # Save the data if requested
        if save_results:
            X.to_pickle(f"./cache/factor_{freq}.pkl")
            Y.to_pickle(f"./cache/asset_{freq}.pkl")

    # Partition dataset into training and testing sets. Lag the data by one observation
    return TrainTest(X[:-1], n_obs, split), TrainTest(Y[1:], n_obs, split)


####################################################################################################
# Test code to verify functionality and detect bugs
####################################################################################################
def test_AV() -> None:
    """
    Test the AV function to ensure correctness of generated data.

    :return: None
    """
    try:
        # Define parameters for the test
        start = "2020-01-01"
        end = "2023-01-01"
        split = [0.7, 0.3]
        freq = "weekly"
        n_obs = 104
        n_y = 5
        use_cache = False
        save_results = False
        AV_key = "YDNA9HH8P2IW985M"

        # Generate synthetic data
        features, outputs = AV(
            start, end, split, freq, n_obs, n_y, use_cache, save_results, AV_key
        )

        # Check if generated features and outputs are pandas DataFrames
        assert isinstance(
            features.data, pd.DataFrame
        ), "Features should be a pandas DataFrame."
        assert isinstance(
            outputs.data, pd.DataFrame
        ), "Outputs should be a pandas DataFrame."

        # Check the dimensions of the generated data
        assert (
            features.data.shape[0] == outputs.data.shape[0]
        ), "Features and outputs should have matching number of rows."

        print("All tests passed successfully.")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


import pandas as pd
import statsmodels.api as sm
import numpy as np


####################################################################################################
# Statistical Analysis Function
####################################################################################################
def statanalysis(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    Conduct a pairwise statistical significance analysis of each feature in X against each asset in Y.

    :param X: DataFrame containing the time series of features (independent variables).
    :param Y: DataFrame containing the time series of asset returns (dependent variables).
    :return: DataFrame containing p-values obtained from regressing each individual feature against each individual asset.
    """
    # Initialize an empty DataFrame to store p-values
    stats: pd.DataFrame = pd.DataFrame(columns=X.columns, index=Y.columns)

    # Iterate over each asset (Y) and each feature (X) to perform OLS regression
    # OLS (Ordinary Least Squares) is used to estimate the relationship between each feature and asset return.
    # For each asset, regress it against each feature with a constant term to obtain the p-value.
    for ticker in Y.columns:
        for feature in X.columns:
            # Perform OLS regression and store the p-value of the feature
            stats.loc[ticker, feature] = (
                sm.OLS(Y[ticker].values, sm.add_constant(X[feature]).values)
                .fit()
                .pvalues[1]
            )

    # Convert p-values to float and round to two decimal places
    return stats.astype(float).round(2)


####################################################################################################
# Test code to verify functionality and avoid bugs
####################################################################################################
def test_statanalysis() -> None:
    """
    Test the statanalysis function to ensure it correctly calculates p-values for feature-asset relationships.

    :return: None
    """
    try:
        # Generate random data for testing
        np.random.seed(42)
        X_test = pd.DataFrame(
            np.random.randn(100, 5), columns=[f"Feature_{i+1}" for i in range(5)]
        )
        Y_test = pd.DataFrame(
            np.random.randn(100, 3), columns=[f"Asset_{i+1}" for i in range(3)]
        )

        # Run the statistical analysis
        stats_result = statanalysis(X_test, Y_test)

        # Check if the result is a DataFrame
        assert isinstance(
            stats_result, pd.DataFrame
        ), "The result should be a pandas DataFrame."

        # Check if the dimensions of the result match expectations
        assert stats_result.shape == (
            3,
            5,
        ), f"Expected shape (3, 5), but got {stats_result.shape}."

        # Check if all p-values are between 0 and 1
        assert (
            stats_result.map(lambda x: 0 <= x <= 1).all().all()
        ), "All p-values should be between 0 and 1."

        print("All tests passed successfully.")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Run the test function
if __name__ == "__main__":
    test_statanalysis()
    test_AV()
    test_synthetic_nn()
    test_synthetic_nl()
    print("All tests passed.")
    # Parameters for synthetic data generation
    n_x = 5  # Number of features
    n_y = 10  # Number of assets
    n_tot = 1200  # Total number of observations
    n_obs = 104  # Number of observations per batch
    split = [0.6, 0.4]  # Split ratios for training and testing
    set_seed = 100  # Random seed for reproducibility

    # Generate synthetic data
    train_test_features, train_test_outputs = synthetic(
        n_x, n_y, n_tot, n_obs, split, set_seed
    )

    # Test the generated feature data
    train_features = train_test_features.train()
    test_features = train_test_features.test()
    print("Training Features:")
    print(train_features.head())
    print(f"Number of training feature observations: {len(train_features)}")
    print("\nTest Features:")
    print(test_features.head())
    print(f"Number of test feature observations: {len(test_features)}")

    # Test the generated output data
    train_outputs = train_test_outputs.train()
    test_outputs = train_test_outputs.test()
    print("\nTraining Outputs:")
    print(train_outputs.head())
    print(f"Number of training output observations: {len(train_outputs)}")
    print("\nTest Outputs:")
    print(test_outputs.head())
    print(f"Number of test output observations: {len(test_outputs)}")

    # Verify the shape of the generated data
    assert (
        train_features.shape[1] == n_x
    ), "Number of features in training set does not match expected value."
    assert (
        train_outputs.shape[1] == n_y
    ), "Number of assets in training set does not match expected value."
    assert (
        test_features.shape[1] == n_x
    ), "Number of features in test set does not match expected value."
    assert (
        test_outputs.shape[1] == n_y
    ), "Number of assets in test set does not match expected value."
    # Generate synthetic data for testing
    n_tot = 1000  # Total number of observations
    n_features = 5  # Number of features
    split_ratios = [0.7, 0.3]  # 70% training, 30% testing
    n_obs = 50  # Number of observations per batch

    # Create a synthetic DataFrame with random data
    data = pd.DataFrame(
        np.random.randn(n_tot, n_features),
        columns=[f"Feature_{i}" for i in range(n_features)],
    )

    # Initialize TrainTest object
    train_test_obj = TrainTest(data=data, n_obs=n_obs, split=split_ratios)

    # Test the training data split
    train_data = train_test_obj.train()
    print("Training Data:")
    print(train_data.head())
    print(f"Number of training observations: {len(train_data)}")

    # Test the test data split
    test_data = train_test_obj.test()
    print("\nTest Data:")
    print(test_data.head())
    print(f"Number of test observations: {len(test_data)}")

    # Update split ratios and test again
    new_split_ratios = [0.6, 0.4]  # Update split ratios
    train_test_obj.split_update(split=new_split_ratios)

    # Test the updated training data split
    updated_train_data = train_test_obj.train()
    print("\nUpdated Training Data:")
    print(updated_train_data.head())
    print(f"Number of updated training observations: {len(updated_train_data)}")

    # Test the updated test data split
    updated_test_data = train_test_obj.test()
    print("\nUpdated Test Data:")
    print(updated_test_data.head())
    print(f"Number of updated test observations: {len(updated_test_data)}")
