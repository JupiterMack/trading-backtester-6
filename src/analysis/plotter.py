import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Optional, Dict, Any

# Configure logging for this module
logger = logging.getLogger(__name__)

class Plotter:
    """
    Generates visualizations (using matplotlib) of backtest performance,
    equity curves, and trade signals.
    """

    def __init__(self):
        """
        Initializes the Plotter with a default matplotlib style.
        """
        # Set a professional and readable matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        logger.info("Plotter initialized with 'seaborn-v0_8-darkgrid' style.")

    def plot_equity_curve(self, equity_curve: pd.Series, benchmark_curve: Optional[pd.Series] = None,
                          title: str = "Equity Curve", save_path: Optional[str] = None) -> None:
        """
        Plots the equity curve of the backtest, optionally comparing it against a benchmark.

        Args:
            equity_curve (pd.Series): A Pandas Series representing the strategy's equity curve over time.
                                       The index should be a DatetimeIndex.
            benchmark_curve (pd.Series, optional): A Pandas Series representing a benchmark's equity curve.
                                                   The index should be a DatetimeIndex. Defaults to None.
            title (str): The title of the plot.
            save_path (str, optional): The file path to save the plot (e.g., 'equity_curve.png').
                                       If None, the plot will be displayed.
        """
        if equity_curve.empty:
            logger.warning("Equity curve data is empty. Cannot plot equity curve.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(equity_curve.index, equity_curve.values, label='Strategy Equity', color='blue', linewidth=2)

        if benchmark_curve is not None and not benchmark_curve.empty:
            # Ensure benchmark curve aligns with strategy curve's time range if desired
            # For simplicity, we plot as is, assuming indices are comparable or will be handled upstream.
            ax.plot(benchmark_curve.index, benchmark_curve.values, label='Benchmark Equity',
                    color='orange', linestyle='--', linewidth=1.5)

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Portfolio Value", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path)
                logger.info(f"Equity curve plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save equity curve plot to {save_path}: {e}")
        else:
            plt.show()
        plt.close(fig) # Close the figure to free up memory

    def plot_price_with_signals(self, price_data: pd.DataFrame, trade_log: pd.DataFrame,
                                indicators: Optional[Dict[str, pd.Series]] = None,
                                title: str = "Price Chart with Trade Signals",
                                save_path: Optional[str] = None) -> None:
        """
        Plots the historical price data along with buy/sell signals and optional indicators.

        Args:
            price_data (pd.DataFrame): DataFrame with historical price data. Must have a 'Close' column
                                       and a DatetimeIndex.
            trade_log (pd.DataFrame): DataFrame with trade logs. Expected columns:
                                      'Date' (datetime), 'Type' ('BUY' or 'SELL'), 'Price'.
            indicators (Dict[str, pd.Series], optional): A dictionary where keys are indicator names
                                                         and values are Pandas Series representing the indicator values.
                                                         Each Series should have a DatetimeIndex. Defaults to None.
            title (str): The title of the plot.
            save_path (str, optional): The file path to save the plot (e.g., 'price_signals.png').
                                       If None, the plot will be displayed.
        """
        if price_data.empty or 'Close' not in price_data.columns:
            logger.warning("Price data is empty or missing 'Close' column. Cannot plot price with signals.")
            return

        # Prepare trade log: ensure 'Date' is datetime and set as index for easy lookup
        if not trade_log.empty:
            if 'Date' in trade_log.columns:
                trade_log['Date'] = pd.to_datetime(trade_log['Date'])
                trade_log = trade_log.set_index('Date')
            else:
                logger.warning("Trade log missing 'Date' column. Cannot plot signals accurately.")
                trade_log = pd.DataFrame() # Clear trade_log if 'Date' is missing

        # Filter trade_log to only include dates present in price_data index
        if not trade_log.empty:
            trade_log = trade_log[trade_log.index.isin(price_data.index)]

        # Determine number of subplots: 1 for price, plus one for each indicator
        num_subplots = 1 + (len(indicators) if indicators else 0)
        fig, axes = plt.subplots(num_subplots, 1, figsize=(14, 6 * num_subplots), sharex=True)

        # Ensure axes is an array even if only one subplot
        if num_subplots == 1:
            axes = [axes]

        # --- Plot Price and Signals on the first subplot ---
        ax_price = axes[0]
        ax_price.plot(price_data.index, price_data['Close'], label='Close Price', color='black', linewidth=1)

        if not trade_log.empty:
            buy_signals = trade_log[trade_log['Type'] == 'BUY']
            sell_signals = trade_log[trade_log['Type'] == 'SELL']

            # Plot buy signals
            if not buy_signals.empty:
                # Use the actual 'Close' price from price_data at the signal date for plotting accuracy
                buy_prices = price_data.loc[buy_signals.index, 'Close']
                ax_price.scatter(buy_signals.index, buy_prices,
                                 marker='^', color='green', s=100, label='Buy Signal', alpha=0.8, zorder=5)
            # Plot sell signals
            if not sell_signals.empty:
                # Use the actual 'Close' price from price_data at the signal date for plotting accuracy
                sell_prices = price_data.loc[sell_signals.index, 'Close']
                ax_price.scatter(sell_signals.index, sell_prices,
                                 marker='v', color='red', s=100, label='Sell Signal', alpha=0.8, zorder=5)

        ax_price.set_title(title, fontsize=16)
        ax_price.set_ylabel("Price", fontsize=12)
        ax_price.legend(fontsize=10)
        ax_price.grid(True, linestyle=':', alpha=0.7)

        # --- Plot Indicators on subsequent subplots ---
        if indicators:
            for i, (indicator_name, indicator_series) in enumerate(indicators.items()):
                ax_indicator = axes[i + 1] # +1 because the first subplot is for price
                if not indicator_series.empty:
                    # Ensure indicator series index aligns with price_data index for shared x-axis
                    indicator_series = indicator_series[indicator_series.index.isin(price_data.index)]
                    ax_indicator.plot(indicator_series.index, indicator_series.values, label=indicator_name, linewidth=1.5)
                    ax_indicator.set_ylabel(indicator_name, fontsize=12)
                    ax_indicator.legend(fontsize=10)
                    ax_indicator.grid(True, linestyle=':', alpha=0.7)
                else:
                    logger.warning(f"Indicator '{indicator_name}' data is empty. Skipping plot.")

        # Set common X-axis label for the last subplot and rotate ticks
        axes[-1].set_xlabel("Date", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        if save_path:
            try:
                plt.savefig(save_path)
                logger.info(f"Price chart with signals plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save price chart with signals plot to {save_path}: {e}")
        else:
            plt.show()
        plt.close(fig)

    def plot_drawdown(self, drawdown_series: pd.Series, title: str = "Drawdown Curve", save_path: Optional[str] = None) -> None:
        """
        Plots the drawdown curve of the backtest.

        Args:
            drawdown_series (pd.Series): A Pandas Series representing the drawdown over time.
                                         The index should be a DatetimeIndex. Values should be negative or zero.
            title (str): The title of the plot.
            save_path (str, optional): The file path to save the plot (e.g., 'drawdown.png').
                                       If None, the plot will be displayed.
        """
        if drawdown_series.empty:
            logger.warning("Drawdown data is empty. Cannot plot drawdown curve.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Fill the area below the drawdown curve for better visualization
        ax.fill_between(drawdown_series.index, drawdown_series.values, 0, color='red', alpha=0.3, label='Drawdown')
        ax.plot(drawdown_series.index, drawdown_series.values, color='red', linewidth=1)

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Drawdown (%)", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(min(drawdown_series.min(), -0.01), 0) # Ensure y-axis starts from 0 or below
        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path)
                logger.info(f"Drawdown plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save drawdown plot to {save_path}: {e}")
        else:
            plt.show()
        plt.close(fig)

# Example usage (for demonstration/testing purposes, not typically run directly in production)
if __name__ == '__main__':
    # Setup basic logging for the example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    plotter = Plotter()

    # --- Dummy Data for Equity Curve ---
    dates_eq = pd.date_range(start='2022-01-01', periods=252, freq='B') # Business days for a year
    equity_data = pd.Series((1 + np.random.randn(252) * 0.01).cumprod() * 1000, index=dates_eq)
    benchmark_data = pd.Series((1 + np.random.randn(252) * 0.005).cumprod() * 1000, index=dates_eq)
    plotter.plot_equity_curve(equity_data, benchmark_data, title="Strategy vs. Benchmark Equity Curve Example")
    # plotter.plot_equity_curve(equity_data, save_path="equity_curve_example.png") # Example save

    # --- Dummy Data for Price with Signals and Indicators ---
    dates_price = pd.date_range(start='2022-01-01', periods=300, freq='D')
    price_close = pd.Series(100 + np.random.randn(300).cumsum() * 0.5 + np.sin(np.arange(300)/10) * 5, index=dates_price)
    price_df = pd.DataFrame({'Close': price_close, 'Open': price_close * 0.99, 'High': price_close * 1.01, 'Low': price_close * 0.98})

    # Dummy trade log (ensure dates are within price_df index)
    trade_log_data = [
        {'Date': pd.Timestamp('2022-01-15'), 'Type': 'BUY', 'Price': price_df.loc['2022-01-15', 'Close']},
        {'Date': pd.Timestamp('2022-02-01'), 'Type': 'SELL', 'Price': price_df.loc['2022-02-01', 'Close']},
        {'Date': pd.Timestamp('2022-02-20'), 'Type': 'BUY', 'Price': price_df.loc['2022-02-20', 'Close']},
        {'Date': pd.Timestamp('2022-03-10'), 'Type': 'SELL', 'Price': price_df.loc['2022-03-10', 'Close']},
        {'Date': pd.Timestamp('2022-04-05'), 'Type': 'BUY', 'Price': price_df.loc['2022-04-05', 'Close']},
        {'Date': pd.Timestamp('2022-04-25'), 'Type': 'SELL', 'Price': price_df.loc['2022-04-25', 'Close']},
    ]
    trade_log_df = pd.DataFrame(trade_log_data)

    # Dummy indicators (align indices with price_df)
    sma_20 = price_df['Close'].rolling(window=20).mean()
    rsi_14 = pd.Series(np.random.rand(300) * 80 + 10, index=dates_price) # Simulate RSI values 10-90
    indicators_dict = {
        'SMA_20': sma_20,
        'RSI_14': rsi_14
    }

    plotter.plot_price_with_signals(price_df, trade_log_df, indicators=indicators_dict,
                                    title="Stock Price with MA, RSI, and Trade Signals Example")
    # plotter.plot_price_with_signals(price_df, trade_log_df, save_path="price_signals_example.png") # Example save

    # --- Dummy Data for Drawdown Curve ---
    # Calculate a simple drawdown from equity curve for demonstration
    peak = equity_data.expanding(min_periods=1).max()
    drawdown_pct = (equity_data - peak) / peak * 100
    plotter.plot_drawdown(drawdown_pct, title="Strategy Drawdown Curve Example")
    # plotter.plot_drawdown(drawdown_pct, save_path="drawdown_example.png") # Example save

    logger.info("All dummy plots generated and displayed (or saved if save_path was provided).")