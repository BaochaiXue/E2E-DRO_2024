import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

####################################################################################################
# SlidingWindow Dataset to index data using a sliding window
####################################################################################################


class SlidingWindow(Dataset):
    """Dataset class for creating a sliding window from time series data."""

    def __init__(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        n_obs: int,
        perf_period: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initialize the SlidingWindow dataset.

        :param X: DataFrame containing the complete feature dataset.
        :param Y: DataFrame containing the complete asset return dataset.
        :param n_obs: Number of observations in the sliding window.
        :param perf_period: Number of future observations used for out-of-sample performance evaluation.
        :param dtype: The desired data type for tensors (default is torch.float32).
        :param device: Device on which to place the tensors (e.g., 'cpu' or 'cuda' for GPU).
        """
        self.X = torch.tensor(
            X.values, dtype=dtype, device=device
        )  # Convert feature dataset to tensor
        self.Y = torch.tensor(
            Y.values, dtype=dtype, device=device
        )  # Convert asset return dataset to tensor
        self.n_obs = n_obs  # Number of observations in the sliding window
        self.perf_period = (
            perf_period  # Number of future observations for performance evaluation
        )

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single window of data.

        :param index: Index of the sliding window.
        :return: Tuple (x, y, y_perf):
            x: Features window of shape (n_obs + 1, n_x).
            y: Realizations window of shape (n_obs, n_y).
            y_perf: Future performance window of shape (perf_period, n_y).
        """
        # Retrieve features for the sliding window (n_obs + 1 observations)
        x = self.X[index : index + self.n_obs + 1]
        # Retrieve asset returns for the sliding window (n_obs observations)
        y = self.Y[index : index + self.n_obs]
        # Retrieve future performance data (perf_period observations)
        y_perf = self.Y[index + self.n_obs : index + self.n_obs + self.perf_period]
        return (x, y, y_perf)

    def __len__(self) -> int:
        """
        Return the number of windows that can be created from the dataset.

        :return: Length of the dataset, considering the sliding windows.
        """
        return (
            len(self.X) - self.n_obs - self.perf_period
        )  # Total number of sliding windows available


####################################################################################################
# Backtest class to store out-of-sample results
####################################################################################################


class Backtest:
    """Class to store out-of-sample results for a backtest."""

    def __init__(self, len_test: int, n_y: int, dates: pd.DatetimeIndex) -> None:
        """
        Initialize the Backtest object.

        :param len_test: Number of scenarios in the out-of-sample evaluation period.
        :param n_y: Number of assets in the portfolio.
        :param dates: DatetimeIndex containing the corresponding dates.
        """
        self.weights = np.zeros(
            (len_test, n_y)
        )  # Initialize portfolio weights over time
        self.rets = np.zeros(
            len_test
        )  # Initialize realized portfolio returns over time
        self.dates = dates[
            -len_test:
        ]  # Keep only the dates for the out-of-sample period

    def stats(self) -> None:
        """
        Compute and store the cumulative returns, mean return, volatility, and Sharpe ratio.

        This method calculates key performance metrics of the portfolio, including:
        - Cumulative returns (Total Return Index), which show the total growth of the portfolio over time.
        - Annualized mean return, which is an estimate of the average return the portfolio would achieve per year.
        - Volatility, which measures the risk by calculating the standard deviation of returns.
        - Sharpe ratio, which indicates the risk-adjusted return of the portfolio.
        """
        # Calculate cumulative returns (Total Return Index)
        tri = np.cumprod(self.rets + 1)
        # Calculate the annualized mean return using the final cumulative return and the number of periods
        self.mean = (tri[-1]) ** (1 / len(tri)) - 1
        # Calculate the volatility (standard deviation) of the portfolio returns
        self.vol = np.std(self.rets)
        # Calculate the Sharpe ratio (mean return divided by volatility)
        self.sharpe = self.mean / self.vol
        # Create a DataFrame containing realized returns and cumulative returns, indexed by dates
        if len(self.dates) == len(self.rets):
            self.rets = pd.DataFrame(
                {"Date": self.dates, "rets": self.rets, "tri": tri}
            ).set_index("Date")
        else:
            raise ValueError("Length of dates and returns must be equal.")


####################################################################################################
# InSample class to store in-sample results
####################################################################################################


class InSample:
    """Class to store the in-sample results of neural network training."""

    def __init__(self) -> None:
        """
        Initialize the InSample object.
        """
        self.loss = []  # List to hold training losses
        self.gamma = []  # List to hold gamma values (hyperparameter)
        self.delta = []  # List to hold delta values (hyperparameter)
        self.val_loss = []  # List to hold validation losses (optional)

    def df(self) -> pd.DataFrame:
        """
        Return a DataFrame containing the training statistics.

        :return: DataFrame with columns representing different metrics during training.
        """
        # Return a DataFrame based on available data, adjusting columns accordingly
        if not self.delta and not self.val_loss:
            return pd.DataFrame(
                list(zip(self.loss, self.gamma)), columns=["loss", "gamma"]
            )
        elif not self.delta:
            return pd.DataFrame(
                list(zip(self.loss, self.val_loss, self.gamma)),
                columns=["loss", "val_loss", "gamma"],
            )
        elif not self.val_loss:
            return pd.DataFrame(
                list(zip(self.loss, self.gamma, self.delta)),
                columns=["loss", "gamma", "delta"],
            )
        else:
            return pd.DataFrame(
                list(zip(self.loss, self.val_loss, self.gamma, self.delta)),
                columns=["loss", "val_loss", "gamma", "delta"],
            )


####################################################################################################
# CrossVal class to store cross-validation results
####################################################################################################


class CrossVal:
    """Class to store cross-validation results of neural network training."""

    def __init__(self) -> None:
        """
        Initialize the CrossVal object.
        """
        self.lr = []  # List to hold learning rates
        self.epochs = []  # List to hold the number of epochs in each run
        self.val_loss = []  # List to hold validation losses

    def df(self) -> pd.DataFrame:
        """
        Return a DataFrame containing the cross-validation statistics.

        :return: DataFrame with learning rate, epochs, and validation loss.
        """
        # Create and return a DataFrame with learning rates, epochs, and validation losses
        return pd.DataFrame(
            list(zip(self.lr, self.epochs, self.val_loss)),
            columns=["lr", "epochs", "val_loss"],
        )


####################################################################################################
# Test code for the Backtest class
####################################################################################################

if __name__ == "__main__":
    # Example usage
    X = pd.DataFrame(
        np.random.randn(100, 3)
    )  # Create feature dataset with 100 samples and 3 features
    Y = pd.DataFrame(
        np.random.randn(100, 2)
    )  # Create asset return dataset with 100 samples and 2 assets
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize SlidingWindow with given parameters
    n_obs = 10
    perf_period = 5
    sliding_window = SlidingWindow(X, Y, n_obs, perf_period, device=device)

    # Fetch a sample window
    print("Testing SlidingWindow...")
    x, y, y_perf = sliding_window[0]
    print(f"x (features): {x.shape}")
    print(f"y (realizations): {y.shape}")
    print(f"y_perf (performance window): {y_perf.shape}")

    # Initialize Backtest with given parameters
    len_test = 30
    backtest_obj = Backtest(len_test=len_test, n_y=2, dates=dates)

    # Simulate some portfolio returns
    backtest_obj.rets = np.random.randn(len_test)

    print("\nTesting Backtest...")
    backtest_obj.stats()
    print(backtest_obj.rets.head())
    print(f"Mean return: {backtest_obj.mean:.4f}")
    print(f"Volatility: {backtest_obj.vol:.4f}")
    print(f"Sharpe ratio: {backtest_obj.sharpe:.4f}")

    # Test Backtest stats calculation
    print("\nTesting Backtest stats calculation...")
    backtest_obj.rets = np.array([0.05, 0.02, 0.03, 0.04, 0.01])
    backtest_obj.dates = dates[
        -len(backtest_obj.rets) :
    ]  # Adjust dates to match returns length
    backtest_obj.stats()
    print(backtest_obj.rets.head())
    print(f"Mean return: {backtest_obj.mean:.4f}")
    print(f"Volatility: {backtest_obj.vol:.4f}")
    print(f"Sharpe ratio: {backtest_obj.sharpe:.4f}")
