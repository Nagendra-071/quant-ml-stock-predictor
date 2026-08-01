import numpy as np
import pandas as pd
import yfinance as yf


def compute_rsi(series, window=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


# end is None to fetch data up to today
def scaled_bse_data(ticker, start="2024-01-01", end=None):

    # Fetch price history using yfinance Ticker API
    df = yf.download(
        ticker, start=start, end=end, auto_adjust=False, progress=False
    )

    # Fallback to alternative ticker if initial data is empty
    if df.empty or len(df) <= 2:
        alt_ticker = "RELIANCE.BO" if ".NS" in ticker else "RELIANCE.NS"
        df = yf.download(
            alt_ticker, start=start, end=end, auto_adjust=False, progress=False
        )

    # Check if dataset is empty
    if df.empty:
        raise ValueError(
            f"No valid trading data found for {ticker}. Check ticker symbol or date range."
        )

    # Flatten 2D columns to 1D arrays to prevent NaN bugs
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Clean price and volume series
    close = df["Close"].ffill().bfill()
    open_p = df["Open"].ffill().bfill()
    high = df["High"].ffill().bfill()
    low = df["Low"].ffill().bfill()
    volume = df["Volume"].ffill().bfill().replace(0, 1.0)

    # Feature Engineering
    log_return = np.log(close / close.shift(1))
    sma_10 = close.rolling(window=10).mean()
    sma_50 = close.rolling(window=50).mean()

    df_features = pd.DataFrame(
        {
            # Base features
            "Volume_Raw": volume,
            "Price_Change": close - open_p,
            "Daily_Return": close.pct_change(fill_method=None),
            "Log_Return": log_return,
            "HL_Spread": (high - low) / close,
            "SMA_Spread": (sma_10 - sma_50) / sma_50,
            "Volatility_20D": log_return.rolling(window=20).std(),
            "RSI_14": compute_rsi(close, window=14),
            # Supervised Target (1 = Tomorrow UP, 0 = Tomorrow DOWN)
            "Target": (log_return.shift(-1) > 0).astype(int),
        },
        index=df.index,
    ).dropna()

    if df_features.empty:
        raise ValueError(
            "Data became empty after calculating features. Check date range length."
        )

    return df_features


if __name__ == "__main__":
    df_features = scaled_bse_data("RELIANCE.NS")
    print(f"Successfully processed {len(df_features)} rows!")
    print(f"Latest Available Date in Dataset: {df_features.index[-1].strftime('%Y-%m-%d')}")
    print("\nFeature Matrix Preview:")
    print(df_features.tail(3))