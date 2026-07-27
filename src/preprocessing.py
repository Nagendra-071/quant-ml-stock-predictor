import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler

# end is None to fetch data up to today
def scaled_bse_data(ticker="RELIANCE.NS", start="2024-01-01", end=None):
    
    # Fetch price history using yfinance Ticker API
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    
    # Fallback to alternative ticker if initial data is empty
    if df.empty or len(df) <= 2:
        alt_ticker = "RELIANCE.BO" if ".NS" in ticker else "RELIANCE.NS"
        df = yf.download(alt_ticker, start=start, end=end, auto_adjust=False, progress=False)

    # Check if dataset is empty
    if df.empty:
        raise ValueError(f"No valid trading data found for {ticker}. Check ticker symbol or date range.")

    # Flatten 2D columns to 1D arrays to prevent NaN bugs
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Handling missing values 
    vol_series = df['Volume'].ffill().bfill().replace(0, 1.0)
    close_series = df['Close'].ffill().bfill()
    open_series = df['Open'].ffill().bfill()


    # Create  features DataFrame
    df_features = pd.DataFrame({
        'Volume_Raw': df['Volume'].ffill().bfill().replace(0, 1.0),
        'Price_Change': df['Close'] - df['Open'],
        'Daily_Return': df['Close'].pct_change(fill_method=None)
    }, index=df.index).dropna()

    if df_features.empty:
        raise ValueError(f"Data became empty after calculating returns. Check column shapes.")

    # StandardScaler
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_features[['Volume_Raw', 'Price_Change', 'Daily_Return']])

    df_scaled = pd.DataFrame(
        scaled_array,
        columns=['Volume_Scaled', 'Price_Change_Scaled', 'Daily_Return_Scaled'],
        index=df_features.index
    )

    return df_features, df_scaled

if __name__ == "__main__":
    raw_df, scaled_df = scaled_bse_data("RELIANCE.NS")
    print(f"Successfully processed {len(scaled_df)} rows!")
    print(f"Latest Available Date in Dataset: {scaled_df.index[-1].strftime('%Y-%m-%d')}")
    print("\nScaled Data Preview:")
    print(scaled_df.tail(3))