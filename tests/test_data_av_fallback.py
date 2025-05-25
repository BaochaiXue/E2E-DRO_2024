import pandas as pd
import numpy as np
from datetime import datetime
import pytest

from e2edro import DataLoad


def test_av_fallback_uses_yfinance(monkeypatch):
    tickers = ["AAPL", "MSFT"]
    dates = pd.date_range("1999-12-30", periods=6, freq="D")
    multi_idx = pd.MultiIndex.from_product([["Adj Close"], tickers])
    fake_prices = pd.DataFrame(
        np.arange(len(dates) * len(tickers)).reshape(len(dates), len(tickers)),
        index=dates,
        columns=multi_idx,
    )

    def fake_download(*args, **kwargs):
        return fake_prices

    def fake_ff(name, start=None, end=None):
        ff_dates = pd.date_range(start, end, freq="D")
        cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
        data = pd.DataFrame(np.random.rand(len(ff_dates), len(cols)), index=ff_dates, columns=cols)
        return {0: data}

    # Patch network calls
    monkeypatch.setattr(DataLoad.yf, "download", fake_download)
    monkeypatch.setattr(DataLoad.pdr, "get_data_famafrench", fake_ff)
    monkeypatch.setattr(DataLoad, "TimeSeries", lambda *a, **k: pytest.fail("TimeSeries should not be called"))

    X, Y = DataLoad.AV(
        "2000-01-01",
        "2000-01-04",
        [0.6, 0.4],
        n_obs=2,
        n_y=2,
        use_cache=False,
        AV_key=None,
    )

    assert Y.data.shape[1] == 2
    assert not Y.data.empty
