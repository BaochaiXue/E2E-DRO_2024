import time
import logging
import pandas as pd
import yfinance as yf


def robust_download(tickers, start, end, max_retries=3, pause=1.0):
    """Download OHLC data via ``yfinance`` with retries and serial requests.

    Parameters
    ----------
    tickers : list[str]
        Symbols to download.
    start : str
        Start date.
    end : str
        End date.
    max_retries : int, optional
        Number of retry attempts for failed tickers, by default 3.
    pause : float, optional
        Seconds to wait between retries, by default 1.0.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted close prices for all successfully downloaded
        tickers. Tickers that still fail after ``max_retries`` will be
        missing from the returned frame.
    """

    remaining = list(tickers)
    collected = []

    for attempt in range(1, max_retries + 1):
        if not remaining:
            break

        df = yf.download(
            remaining,
            start=start,
            end=end,
            progress=False,
            group_by="ticker",
            threads=False,
            auto_adjust=False,
            repair=True,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df = df["Adj Close"]

        good = [t for t in remaining if t in df.columns and not df[t].isna().all()]
        bad = [t for t in remaining if t not in good]

        collected.append(df[good])
        remaining = bad

        if remaining:
            logging.warning(
                "Retry %d/%d \u2013 still missing: %s", attempt, max_retries, remaining
            )
            time.sleep(pause)

    if remaining:
        logging.error("Final failures after retries: %s", remaining)

    out = pd.concat(collected, axis=1).dropna(how="all")
    return out
