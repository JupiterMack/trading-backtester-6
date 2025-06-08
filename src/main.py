import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta
import configparser
import os
from datetime import datetime

# --- Configuration Loading ---
def load_config(config_path='config.ini'):
    """
    Loads configuration from config.ini.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        configparser.ConfigParser: Loaded configuration object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'. Please create it.")
    config.read(config_path)
    return config

# --- Data Loading ---
def load_historical_data(ticker, start_date, end_date, data_source, local_csv_path=None):
    """
    Loads historical stock data from the specified source.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        data_source (str): 'yfinance' or 'local_csv'.
        local_csv_path (str, optional): Path to local CSV if data_source is 'local_csv'.

    Returns:
        pd.DataFrame: DataFrame with historical data (Open, High, Low, Close, Volume).
                      Returns None if data loading fails.
    """
    print(f"Loading data for {ticker} from {start_date} to {end_date} using {data_source}...")
    if data_source == 'yfinance':
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                print(f"Warning: No data downloaded for {ticker}. Check ticker, dates, or internet connection.")
                return None
            # yfinance provides 'Adj Close', which is usually preferred for backtesting.
            # Ensure 'Close' column exists, potentially using 'Adj Close'.
            if 'Adj Close' in data.columns:
                data['Close'] = data['Adj Close']
            return data[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            print(f"Error downloading data from yfinance: {e}")
            return None
    elif data_source == 'local_csv':
        if not local_csv_path or not os.path.exists(local_csv_path):
            print(f"Error: Local CSV path not provided or file not found: {local_csv_path}")
            return None
        try:
            # Assuming CSV has 'Date' column and standard OHLCV columns
            data = pd.read_csv(local_csv_path, index_col='Date', parse_dates=True)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                print(f"Error: Local CSV missing required columns. Found: {data.columns.tolist()}")
                print(f"Expected: {required_cols}")
                return None
            return data[required_cols]
        except Exception as e:
            print(f"Error loading data from local CSV: {e}")
            return None
    else:
        print(f"Error: Unknown data source '{data_source}'. Supported: 'yfinance', 'local_csv'.")
        return None

# --- Strategy Implementation (Simple Moving Average Crossover) ---
def apply_strategy(data, short_window=20, long_window=50):
    """
    Applies a Simple Moving Average (SMA) crossover strategy to the data.

    Generates 'signal' column:
    1 for buy (short MA crosses above long MA)
    -1 for sell (short MA crosses below long MA)
    0 for hold

    Args:
        data (pd.DataFrame): Historical stock data with a 'Close' column.
        short_window (int): Window for the short-term moving average.
        long_window (int): Window for the long-term moving average.

    Returns:
        pd.DataFrame: Data with 'SMA_Short', 'SMA_Long', and 'signal' columns.
                      Returns an empty DataFrame if input data is invalid.
    """
    if data is None or data.empty or 'Close' not in data.columns:
        print("Invalid data for strategy application. Missing 'Close' column or data is empty.")
        return pd.DataFrame()

    df = data.copy()

    # Calculate SMAs using pandas_ta
    df['SMA_Short'] = ta.sma(df['Close'], length=short_window)
    df['SMA_Long'] = ta.sma(df['Close'], length=long_window)

    # Generate signals
    df['signal'] = 0
    # A buy signal occurs when the short MA crosses above the long MA
    df.loc[df['SMA_Short'] > df['SMA_Long'], 'signal'] = 1
    # A sell signal occurs when the short MA crosses below the long MA
    df.loc[df['SMA_Short'] < df['SMA_Long'], 'signal'] = -1

    # Shift signal to prevent look-ahead bias: signal generated at close of day t, acted upon at open of day t+1.
    # For simplicity in this backtester, we assume action is taken at the close of the signal day.
    # For a more robust backtester, consider `df['signal'] = df['signal'].shift(1)` and acting on the next day's open.
    
    return df

# --- Backtesting Engine ---
def run_backtest(data, initial_capital=100000, transaction_cost_pct=0.001):
    """
    Runs a simple backtest based on strategy signals.

    This backtester assumes full position sizing (buy all available shares, sell all held shares).
    It processes signals day by day and updates portfolio value.

    Args:
        data (pd.DataFrame): Data with 'Close' and 'signal' columns.
        initial_capital (float): Starting capital for the backtest.
        transaction_cost_pct (float): Percentage cost per trade (e.g., 0.001 for 0.1%).

    Returns:
        pd.DataFrame: DataFrame containing daily portfolio metrics (cash, shares, holdings, total_value, trades).
                      Returns an empty DataFrame if input data is invalid.
    """
    if data is None or data.empty or 'signal' not in data.columns or 'Close' not in data.columns:
        print("Invalid data for backtesting. Missing 'signal' or 'Close' column, or data is empty.")
        return pd.DataFrame()

    # Initialize portfolio tracking DataFrame
    portfolio = pd.DataFrame(index=data.index)
    portfolio['cash'] = initial_capital
    portfolio['shares'] = 0
    portfolio['holdings'] = 0.0
    portfolio['total_value'] = initial_capital
    portfolio['trades'] = 0 # Counter for executed trades

    in_position = False # Flag to track if we currently hold shares

    # Iterate through the data day by day
    for i in range(len(data)):
        current_date = data.index[i]
        signal = data.loc[current_date, 'signal']
        close_price = data.loc[current_date, 'Close']

        # Get previous day's state (or initial state for the first day)
        if i > 0:
            prev_date = data.index[i-1]
            current_cash = portfolio.loc[prev_date, 'cash']
            current_shares = portfolio.loc[prev_date, 'shares']
            current_trades = portfolio.loc[prev_date, 'trades']
            in_position = (current_shares > 0)
        else:
            current_cash = initial_capital
            current_shares = 0
            current_trades = 0
            in_position = False

        # --- Execute Trades Based on Signal ---
        # Buy signal: If signal is 1 and not already in position
        if signal == 1 and not in_position:
            # Buy with all available cash
            shares_to_buy = int(current_cash / close_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * close_price
                transaction_cost = cost * transaction_cost_pct
                current_cash -= (cost + transaction_cost)
                current_shares += shares_to_buy
                in_position = True
                current_trades += 1
                # print(f"{current_date.strftime('%Y-%m-%d')}: BUY {shares_to_buy} shares at {close_price:.2f}. Cash: {current_cash:.2f}")

        # Sell signal: If signal is -1 and currently in position
        elif signal == -1 and in_position:
            # Sell all shares
            revenue = current_shares * close_price
            transaction_cost = revenue * transaction_cost_pct
            current_cash += (revenue - transaction_cost)
            # print(f"{current_date.strftime('%Y-%m-%d')}: SELL {current_shares} shares at {close_price:.2f}. Cash: {current_cash:.2f}")
            current_shares = 0
            in_position = False
            current_trades += 1
        
        # --- Update Portfolio State for the current day ---
        portfolio.loc[current_date, 'cash'] = current_cash
        portfolio.loc[current_date, 'shares'] = current_shares
        portfolio.loc[current_date, 'holdings'] = current_shares * close_price
        portfolio.loc[current_date, 'total_value'] = current_cash + (current_shares * close_price)
        portfolio.loc[current_date, 'trades'] = current_trades

    return portfolio

# --- Analysis and Visualization ---
def analyze_results(portfolio, initial_capital):
    """
    Analyzes backtest results and prints key performance metrics.

    Args:
        portfolio (pd.DataFrame): DataFrame containing portfolio values over time.
        initial_capital (float): The initial capital used in the backtest.
    """
    if portfolio.empty:
        print("No portfolio data to analyze.")
        return

    final_value = portfolio['total_value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100

    # Calculate daily returns for drawdown and other metrics
    portfolio['daily_return'] = portfolio['total_value'].pct_change().fillna(0) # Fillna for first day
    
    # Max Drawdown: Measures the largest drop from a peak to a trough in the portfolio's value.
    running_max = portfolio['total_value'].cummax()
    drawdown = (portfolio['total_value'] - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    print("\n--- Backtest Results Summary ---")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Total Trades Executed: {portfolio['trades'].iloc[-1]}")
    print("------------------------------")

def plot_results(data, portfolio, ticker):
    """
    Plots the stock price, moving averages, buy/sell signals, and portfolio value.

    Args:
        data (pd.DataFrame): Original data with 'Close', 'SMA_Short', 'SMA_Long', and 'signal'.
        portfolio (pd.DataFrame): Portfolio data with 'total_value'.
        ticker (str): Stock ticker for plot title.
    """
    if data.empty or portfolio.empty:
        print("No data or portfolio to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Price, SMAs, Signals
    ax1.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.7)
    if 'SMA_Short' in data.columns and 'SMA_Long' in data.columns:
        ax1.plot(data.index, data['SMA_Short'], label=f'SMA {data["SMA_Short"].name.split("_")[-1]}', color='orange', linestyle='--')
        ax1.plot(data.index, data['SMA_Long'], label=f'SMA {data["SMA_Long"].name.split("_")[-1]}', color='green', linestyle='--')

    # Plot buy signals
    buy_signals = data[data['signal'] == 1]
    ax1.plot(buy_signals.index, data['Close'].loc[buy_signals.index], '^', markersize=10, color='green', lw=0, label='Buy Signal')

    # Plot sell signals
    sell_signals = data[data['signal'] == -1]
    ax1.plot(sell_signals.index, data['Close'].loc[sell_signals.index], 'v', markersize=10, color='red', lw=0, label='Sell Signal')

    ax1.set_title(f'{ticker} Price and SMA Crossover Strategy Signals')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Portfolio Value
    ax2.plot(portfolio.index, portfolio['total_value'], label='Portfolio Value', color='purple')
    ax2.set_title('Portfolio Value Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting algorithmic trading backtester...")

    # 1. Load Configuration
    try:
        config = load_config()
    except FileNotFoundError as e:
        print(e)
        print("Please ensure 'config.ini' exists in the project root directory or provide its full path.")
        exit(1) # Exit with an error code

    # Get parameters from config
    try:
        data_source = config.get('DATA', 'data_source')
        local_csv_path = config.get('DATA', 'local_csv_path', fallback=None) # Optional path for local CSV

        ticker = config.get('STRATEGY', 'ticker')
        start_date = config.get('STRATEGY', 'start_date')
        end_date = config.get('STRATEGY', 'end_date')
        initial_capital = config.getfloat('STRATEGY', 'initial_capital')
        strategy_name = config.get('STRATEGY', 'strategy_name', fallback='SMA_Crossover') # Default strategy
        short_window = config.getint('STRATEGY', 'short_window', fallback=20)
        long_window = config.getint('STRATEGY', 'long_window', fallback=50)
        
        transaction_cost_pct = config.getfloat('BACKTEST', 'transaction_cost_pct', fallback=0.001)

    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"Configuration Error: Missing section or option in config.ini: {e}.")
        print("Please ensure 'DATA', 'STRATEGY', and 'BACKTEST' sections are correctly defined.")
        exit(1)
    except ValueError as e:
        print(f"Configuration Error: Invalid value type in config.ini: {e}.")
        print("Ensure 'initial_capital', 'short_window', 'long_window', 'transaction_cost_pct' are numbers.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while parsing config.ini: {e}")
        exit(1)

    # 2. Load Historical Data
    historical_data = load_historical_data(ticker, start_date, end_date, data_source, local_csv_path)

    if historical_data is None or historical_data.empty:
        print("Failed to load historical data. Exiting.")
        exit(1)

    # 3. Apply Strategy
    # This section can be extended to dynamically load different strategies based on `strategy_name`.
    # For now, it defaults to SMA Crossover.
    if strategy_name.lower() == 'sma_crossover':
        processed_data = apply_strategy(historical_data, short_window=short_window, long_window=long_window)
    else:
        print(f"Warning: Strategy '{strategy_name}' is not yet implemented. Using default SMA Crossover.")
        processed_data = apply_strategy(historical_data, short_window=short_window, long_window=long_window)

    if processed_data.empty:
        print("Strategy application failed or resulted in empty data. Exiting.")
        exit(1)

    # Remove NaN values introduced by SMA calculation (at the beginning of the series)
    processed_data.dropna(inplace=True)
    if processed_data.empty:
        print("No valid data points remaining after dropping NaNs (likely due to short data range or large SMA windows). Exiting.")
        exit(1)

    # 4. Run Backtest
    portfolio_results = run_backtest(processed_data, initial_capital, transaction_cost_pct)

    if portfolio_results.empty:
        print("Backtest failed or resulted in empty portfolio data. Exiting.")
        exit(1)

    # 5. Analyze and Visualize Results
    analyze_results(portfolio_results, initial_capital)
    plot_results(processed_data, portfolio_results, ticker)

    print("Backtesting complete.")