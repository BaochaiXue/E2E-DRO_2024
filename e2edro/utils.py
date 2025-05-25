import time
import logging
import pandas as pd
import yfinance as yf


def robust_download(tickers, start, end, *, max_retries=3, pause=1.0):
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
        tickers. Columns are always the ticker symbols. Tickers that still
        fail after ``max_retries`` will be missing from the returned frame.
    """

    remaining: list[str] = list(tickers)
    collected: list[pd.DataFrame] = []

    for attempt in range(1, max_retries + 1):
        if not remaining:
            break

        raw = yf.download(
            remaining,
            start=start,
            end=end,
            progress=False,
            group_by="ticker",
            threads=False,
            auto_adjust=False,
            repair=True,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            if "Adj Close" in raw.columns.get_level_values(0):
                prices = raw.xs("Adj Close", level=0, axis=1)
            else:
                prices = raw.xs("Adj Close", level=1, axis=1)
        else:
            prices = raw[["Adj Close"]] if "Adj Close" in raw else raw
            prices.columns = [remaining[0]]

        good = [t for t in remaining if t in prices and not prices[t].isna().all()]
        bad = [t for t in remaining if t not in good]

        collected.append(prices[good])
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
