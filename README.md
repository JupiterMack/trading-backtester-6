# Algorithmic Trading Backtester

A Python tool to backtest stock trading strategies on historical data.

## Overview

This project provides a robust and extensible framework for backtesting algorithmic trading strategies against historical stock market data. Designed with modularity in mind, it allows users to easily define new trading strategies, load data from various sources, simulate trades, manage a virtual portfolio, and analyze performance with key metrics and visualizations.

## Features

*   **Flexible Data Handling**: Load historical stock data from Yahoo Finance or local CSV files.
*   **Modular Strategy Design**: Implement and test various trading strategies (e.g., Moving Averages, RSI) through a clear Abstract Base Class interface, making it easy to add new strategies.
*   **Comprehensive Backtesting Engine**: Simulate trades, manage a virtual portfolio with cash and positions, and apply basic risk management rules.
*   **Detailed Performance Analysis**: Calculate key performance indicators such as total return, CAGR, volatility, Sharpe ratio, and maximum drawdown.
*   **Visualization Tools**: Generate insightful plots of equity curves, trade signals, and other relevant metrics to understand strategy performance.

## Technologies Used

*   Python 3.9+
*   [pandas](https://pandas.pydata.org/) - Data manipulation and analysis
*   [numpy](https://numpy.org/) - Numerical computing
*   [matplotlib](https://matplotlib.org/) - Plotting and visualization
*   [yfinance](https://pypi.org/project/yfinance/) - Download market data from Yahoo! Finance
*   [pandas_ta](https://pypi.org/project/pandas-ta/) - Technical analysis indicators

## Installation

Follow these steps to set up the project locally:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/JupiterMack/trading-backtester-6.git
    cd trading-backtester-6
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the backtester:

1.  **Configure `config.ini`**:
    Before running, open `config.ini` and adjust the parameters according to your needs. You can specify:
    *   `data_source`: `yfinance` or `local_csv`
    *   `symbol`: The stock ticker (e.g., `AAPL`)
    *   `start_date`, `end_date`: Date range for backtesting
    *   `initial_capital`: Starting capital for the portfolio
    *   `strategy_name`: Which strategy to use (e.g., `MovingAverageStrategy`, `RSIStrategy`)
    *   Strategy-specific parameters (e.g., `fast_window`, `slow_window` for Moving Average, `rsi_period`, `rsi_buy_threshold` for RSI).
    *   Risk management parameters (e.g., `position_size_percent`, `stop_loss_percent`).

2.  **Run the main application**:
    ```bash
    python src/main.py
    ```

The backtester will execute the strategy based on your `config.ini` settings, simulate trades, print performance metrics to the console, and generate plots in the `reports/plots` directory (if configured).

## Key Components / Project Structure

*   `README.md`: Comprehensive project overview, setup instructions, and usage guide for the backtester.
*   `LICENSE`: Specifies the licensing terms under which the project's code is distributed.
*   `pyproject.toml`: Modern Python project configuration file, defining project metadata and build system settings.
*   `config.ini`: Configuration file for various project settings, such as data paths, backtest parameters, and strategy specific options.
*   `src/__init__.py`: Makes the 'src' directory a Python package, allowing its modules to be imported.
*   `src/main.py`: The main entry point for running the algorithmic trading backtester, orchestrating data loading, strategy execution, and analysis.
*   `src/data/__init__.py`: Initializes the data handling subpackage.
*   `src/data/data_loader.py`: Handles loading historical stock data from various sources (e.g., CSV, API) into pandas DataFrames.
*   `src/data/data_processor.py`: Contains functions for cleaning, transforming, and preparing raw historical data for backtesting.
*   `src/strategies/__init__.py`: Initializes the trading strategies subpackage.
*   `src/strategies/base_strategy.py`: Defines an abstract base class or interface for all trading strategies, outlining required methods.
*   `src/strategies/moving_average.py`: An example implementation of a trading strategy based on moving averages.
*   `src/strategies/rsi_strategy.py`: An example implementation of a trading strategy based on the Relative Strength Index (RSI).
*   `src/backtester/__init__.py`: Initializes the core backtesting engine subpackage.
*   `src/backtester/engine.py`: The core simulation engine that iterates through historical data, executes trades, and manages the backtest flow.
*   `src/backtester/portfolio_manager.py`: Manages the simulated portfolio's state, including cash, positions, and trade history.
*   `src/backtester/risk_manager.py`: Handles risk management aspects such as position sizing, stop-loss, and take-profit orders.
*   `src/analysis/__init__.py`: Initializes the performance analysis subpackage.
*   `src/analysis/performance_metrics.py`: Calculates key performance indicators (KPIs) like Sharpe ratio, drawdown, and win rate from backtest results.
*   `src/analysis/plotter.py`: Generates visualizations (using matplotlib) of backtest performance, equity curves, and trade signals.
*   `src/utils/__init__.py`: Initializes the general utility functions subpackage.
*   `src/utils/helpers.py`: Contains miscellaneous helper functions used across different parts of the project.
*   `tests/__init__.py`: Makes the 'tests' directory a Python package for test discovery.
*   `tests/test_data_loader.py`: Unit tests for the data loading functionalities.
*   `tests/test_backtester_engine.py`: Unit tests for the core backtesting engine's logic and calculations.
*   `tests/test_strategies.py`: Unit tests for various trading strategy implementations.

## License

This project is licensed under the MIT License.