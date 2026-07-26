import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler

# end is None to fetch data up to today
def scaled_bse_data(ticker="RELIANCE.NS", start="2024-01-01", end=None):
    
    # Fetch price history using yfinance Ticker API
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end, auto_adjust=False)
    
    # Fallback to alternative ticker if initial data is empty
    if df.empty or len(df) <= 2:
        alt_ticker = "RELIANCE.BO" if ".NS" in ticker else "RELIANCE.NS"
        stock = yf.Ticker(alt_ticker)
        df = stock.history(start=start, end=end, auto_adjust=False)

    # Check if dataset is empty
    if df.empty:
        raise ValueError(f"No valid trading data found for {ticker}. Check ticker symbol or date range.")

    # Flatten 2D columns to 1D arrays to prevent NaN bugs
    close_vals = df['Close'].values.flatten().astype(float)
    open_vals = df['Open'].values.flatten().astype(float)
    vol_vals = df['Volume'].values.flatten().astype(float)

    # Handling missing values 
    close_series = pd.Series(close_vals, index=df.index).ffill().bfill()
    open_series = pd.Series(open_vals, index=df.index).ffill().bfill()
    vol_series = pd.Series(vol_vals, index=df.index).ffill().bfill().replace(0, 1.0)

    # Feature engineering
    price_change = close_series - open_series
    daily_return = close_series.pct_change(fill_method=None)

    # Create clean features DataFrame
    df_features = pd.DataFrame({
        'Volume_Raw': vol_series,
        'Price_Change': price_change,
        'Daily_Return': daily_return
    }, index=df.index)

  
    df_features = df_features.iloc[1:]

  
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