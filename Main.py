import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Check matplotlib version
print("Matplotlib version:", plt.matplotlib.__version__)

def fetch_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data

def calculate_rsi(data, window=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = data['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_stock_data(data, ticker):
    """Plot stock price with moving averages, Bollinger Bands, volume, and RSI."""
    print("Data columns:", data.columns)
    print("Volume dtype:", data['Volume'].dtypes)
    print("Index type:", type(data.index))
    
    # Set up three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Price plot with moving averages and Bollinger Bands
    ax1.plot(data['Adj Close'], label='Adj Close', color='blue')
    ax1.plot(data['MA50'], label='50-day MA', color='orange')
    ax1.plot(data['MA200'], label='200-day MA', color='red')
    ax1.plot(data['upper_band'], color='gray', linestyle='--', alpha=0.5, label='Upper Bollinger Band')
    ax1.plot(data['lower_band'], color='gray', linestyle='--', alpha=0.5, label='Lower Bollinger Band')
    ax1.set_title(f'{ticker} Stock Price with Moving Averages and Bollinger Bands')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid()
    
    # Volume plot
    ax2.bar(data.index, data['Volume'], color='gray', edgecolor='none')
    ax2.set_title('Trading Volume')
    ax2.set_ylabel('Volume')
    
    # RSI plot
    ax3.plot(data.index, data['RSI'], label='RSI', color='purple')
    ax3.axhline(30, linestyle='--', alpha=0.5, color='red')
    ax3.axhline(70, linestyle='--', alpha=0.5, color='red')
    ax3.set_title('Relative Strength Index (RSI)')
    ax3.set_ylabel('RSI')
    ax3.set_xlabel('Date')
    
    plt.tight_layout()
    plt.show()

def display_metrics(stock_data, ticker, start_date, end_date):
    index_ticker = '^GSPC'  # S&P 500 index
    
    # Download index data
    index_data = yf.download(index_ticker, start=start_date, end=end_date, auto_adjust=False)
    if index_data.empty:
        print(f"No data found for {index_ticker} between {start_date} and {end_date}.")
        correlation = np.nan
    else:
        index_returns = index_data['Adj Close'].pct_change().dropna()
        stock_returns = stock_data['Adj Close'].pct_change().dropna()
        
        # Check data sufficiency and compute correlation
        try:
            if len(stock_returns) < 2 or len(index_returns) < 2:
                print(f"Insufficient data points for correlation between {ticker} and {index_ticker}.")
                correlation = np.nan
            else:
                correlation = stock_returns.corr(index_returns)
        except IndexError as e:
            print(f"Error computing correlation: {e}. Setting correlation to NaN.")
            correlation = np.nan
        
        # Display the result
        print(f"Correlation with {index_ticker}: {correlation}")

def ml_forecast(data, ticker, window_size=30, future_days=10):
    """Perform ML-based forecasting."""
    print(f"Running forecast with window size: {window_size}")
    
    for i in range(window_size):
        data[f'lag_{i+1}'] = data['Adj Close'].shift(i + 1)
    
    feature_cols = [f'lag_{i+1}' for i in range(window_size)]
    data.dropna(subset=feature_cols + ['Adj Close'], inplace=True)
    
    print("After lag features:")
    print("Data shape:", data.shape)
    
    X = data[feature_cols]
    y = data['Adj Close']
    
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(window_size,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    last_sequence = data['Adj Close'].iloc[-window_size:].values
    future_preds = []
    for _ in range(future_days):
        input_seq = last_sequence[-window_size:]
        input_seq_scaled = scaler_X.transform(input_seq.reshape(1, -1))
        next_pred_scaled = model.predict(input_seq_scaled, verbose=0)[0][0]
        next_pred = scaler_y.inverse_transform([[next_pred_scaled]])[0][0]
        future_preds.append(next_pred)
        last_sequence = np.append(last_sequence, next_pred)
    
    future_dates = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
    
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Adj Close'], label='Actual', color='blue')
    plt.plot(X_test.index, y_pred, label='Predicted (test)', color='orange')
    plt.plot(future_dates, future_preds, label='Future Predictions', color='green')
    plt.title(f'{ticker} Stock Price - Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()
    plt.show()
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nML Model Performance for {ticker}:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

if __name__ == "__main__":
    ticker = input("Enter a single stock ticker symbol (e.g., AAPL): ").strip()
    if len(ticker.split()) > 1:
        print("Please enter only one ticker symbol.")
        import sys
        sys.exit(1)
    
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    
    stock_data = fetch_data(ticker, start_date, end_date)
    
    if stock_data.empty:
        print(f"No data found for {ticker} between {start_date} and {end_date}.")
    else:
        # Add moving averages, Bollinger Bands, and RSI
        stock_data['MA50'] = stock_data['Adj Close'].rolling(window=50).mean()
        stock_data['MA200'] = stock_data['Adj Close'].rolling(window=200).mean()
        stock_data['MA20'] = stock_data['Adj Close'].rolling(window=20).mean()
        stock_data['std20'] = stock_data['Adj Close'].rolling(window=20).std()
        stock_data['upper_band'] = stock_data['MA20'] + 2 * stock_data['std20']
        stock_data['lower_band'] = stock_data['MA20'] - 2 * stock_data['std20']
        stock_data['RSI'] = calculate_rsi(stock_data)
        
        plot_stock_data(stock_data, ticker)
        display_metrics(stock_data, ticker, start_date, end_date)
        
        ml_choice = input("Do you want to perform ML forecasting? (yes/no): ").strip().lower()
        if ml_choice == 'yes':
            ml_forecast(stock_data, ticker, window_size=10)