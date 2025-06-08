import pandas as pd
import pandas_ta as ta
import logging
import numpy as np
import matplotlib.pyplot as plt
from src.strategies.base_strategy import BaseStrategy

# Configure logging for this module
logger = logging.getLogger(__name__)

class RSIStrategy(BaseStrategy):
    """
    An example implementation of a trading strategy based on the Relative Strength Index (RSI).

    This strategy generates buy signals when the RSI crosses above the oversold level
    (after being below it) and sell signals when the RSI crosses below the overbought level
    (after being above it).
    """

    def __init__(self, rsi_period: int = 14, overbought_level: int = 70, oversold_level: int = 30):
        """
        Initializes the RSI Strategy with specified parameters.

        Args:
            rsi_period (int): The number of periods to use for RSI calculation.
            overbought_level (int): The RSI level considered overbought, triggering a potential sell signal.
            oversold_level (int): The RSI level considered oversold, triggering a potential buy signal.
        """
        super().__init__("RSI Strategy")
        if not (1 <= rsi_period <= 200): # Reasonable range for period
            raise ValueError("RSI period must be between 1 and 200.")
        if not (0 <= oversold_level < overbought_level <= 100):
            raise ValueError("Oversold level must be less than overbought level, and both between 0 and 100.")

        self.rsi_period = rsi_period
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        logger.info(f"RSI Strategy initialized with period={self.rsi_period}, "
                    f"overbought={self.overbought_level}, oversold={self.oversold_level}")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates raw buy and sell signals based on RSI crossovers.

        Args:
            data (pd.DataFrame): Historical price data, must contain a 'Close' column.

        Returns:
            pd.DataFrame: The input DataFrame with added 'buy_signal' and 'sell_signal' boolean columns.
                          These signals indicate potential entry/exit points before applying full strategy logic.
        """
        if 'Close' not in data.columns:
            logger.error("Data must contain a 'Close' column for RSI calculation.")
            raise ValueError("Missing 'Close' column in data for RSI strategy.")

        # Ensure data is sorted by index (typically datetime) for correct indicator calculation
        data = data.sort_index()

        # Calculate RSI using pandas_ta
        # The column name will be like 'RSI_14' if period is 14
        rsi_col_name = f"RSI_{self.rsi_period}"
        data[rsi_col_name] = ta.rsi(data['Close'], length=self.rsi_period)

        # Initialize signal columns
        data['buy_signal'] = False
        data['sell_signal'] = False

        # Generate buy signal: RSI crosses above oversold level from below
        # This means RSI was below oversold in the previous period and is at or above it now.
        data.loc[(data[rsi_col_name].shift(1) < self.oversold_level) &
                 (data[rsi_col_name] >= self.oversold_level), 'buy_signal'] = True

        # Generate sell signal: RSI crosses below overbought level from above
        # This means RSI was above overbought in the previous period and is at or below it now.
        data.loc[(data[rsi_col_name].shift(1) > self.overbought_level) &
                 (data[rsi_col_name] <= self.overbought_level), 'sell_signal'] = True

        # Fill NaN values (which occur at the beginning due to indicator calculation) with False
        data['buy_signal'] = data['buy_signal'].fillna(False)
        data['sell_signal'] = data['sell_signal'].fillna(False)

        logger.debug(f"Generated RSI signals for {len(data)} data points. "
                     f"Buy signals: {data['buy_signal'].sum()}, Sell signals: {data['sell_signal'].sum()}")
        return data

    def _apply_strategy_logic(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the sequential trading logic based on generated signals.

        This method determines the actual 'buy', 'sell', or 'hold' actions
        considering the current position (e.g., preventing multiple buys without a sell).

        Args:
            data (pd.DataFrame): The DataFrame containing 'Close', 'buy_signal', and 'sell_signal' columns.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'action' column ('buy', 'sell', 'hold').
        """
        data['action'] = 'hold'
        position = 0  # 0: no position, 1: long position

        # Iterate through the DataFrame to apply sequential logic.
        # While loops can be slower than vectorized operations for large datasets,
        # they are often necessary and clearer for stateful, sequential trading logic.
        for i in range(len(data)):
            current_date = data.index[i]
            buy_signal = data.loc[current_date, 'buy_signal']
            sell_signal = data.loc[current_date, 'sell_signal']

            if buy_signal and position == 0:
                # If a buy signal is present and we are not currently holding a position
                data.loc[current_date, 'action'] = 'buy'
                position = 1
                logger.debug(f"[{current_date.strftime('%Y-%m-%d')}] Buy signal triggered. Entering long position.")
            elif sell_signal and position == 1:
                # If a sell signal is present and we are currently holding a position
                data.loc[current_date, 'action'] = 'sell'
                position = 0
                logger.debug(f"[{current_date.strftime('%Y-%m-%d')}] Sell signal triggered. Exiting long position.")
            else:
                # Otherwise, hold the current position
                data.loc[current_date, 'action'] = 'hold'
                # If position is 1 and no sell signal, continue holding.
                # If position is 0 and no buy signal, continue holding (no position).
                pass

        logger.info(f"Applied RSI strategy logic to {len(data)} data points. "
                    f"Total buys: {data[data['action'] == 'buy'].shape[0]}, "
                    f"Total sells: {data[data['action'] == 'sell'].shape[0]}.")
        return data

    def plot_strategy(self, data: pd.DataFrame, ax=None):
        """
        Plots the Close price with buy/sell signals and the RSI indicator.

        Args:
            data (pd.DataFrame): The DataFrame containing 'Close' prices, 'action' signals,
                                 and the calculated RSI column.
            ax (matplotlib.axes.Axes, optional): An existing Axes object to plot the price on.
                                                 If None, a new figure and axes are created.
                                                 A second subplot for RSI will always be created.
        """
        if 'Close' not in data.columns or 'action' not in data.columns:
            logger.error("Data must contain 'Close' and 'action' columns for plotting.")
            return

        rsi_col_name = f"RSI_{self.rsi_period}"
        if rsi_col_name not in data.columns:
            logger.warning(f"RSI column '{rsi_col_name}' not found. Recalculating for plot.")
            data = self.generate_signals(data.copy()) # Recalculate if missing

        # Create figure and subplots if no axes are provided
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                           gridspec_kw={'height_ratios': [3, 1]})
        else:
            # If an ax is provided, it's for the price chart. Create a new subplot for RSI below it.
            fig = ax.figure
            gs = ax.get_gridspec()
            ax1 = ax
            # Add new subplot for RSI below the provided ax1
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

        # --- Plot Price Chart with Signals (ax1) ---
        ax1.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=1)

        # Plot Buy/Sell signals
        buy_points = data[data['action'] == 'buy']
        sell_points = data[data['action'] == 'sell']

        if not buy_points.empty:
            ax1.scatter(buy_points.index, buy_points['Close'], marker='^', color='green',
                        s=100, label='Buy Signal', zorder=5, alpha=0.8)
        if not sell_points.empty:
            ax1.scatter(sell_points.index, sell_points['Close'], marker='v', color='red',
                        s=100, label='Sell Signal', zorder=5, alpha=0.8)

        ax1.set_ylabel('Price')
        ax1.set_title(f'{self.name} - Price Chart with Trading Signals')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # --- Plot RSI Indicator (ax2) ---
        ax2.plot(data.index, data[rsi_col_name], label=rsi_col_name, color='purple', linewidth=1)
        ax2.axhline(self.overbought_level, linestyle='--', color='red', label='Overbought Level')
        ax2.axhline(self.oversold_level, linestyle='--', color='green', label='Oversold Level')
        ax2.axhline(50, linestyle=':', color='gray', alpha=0.7, label='Midline')

        # Highlight overbought/oversold regions
        ax2.fill_between(data.index, self.overbought_level, 100, color='red', alpha=0.1)
        ax2.fill_between(data.index, 0, self.oversold_level, color='green', alpha=0.1)

        ax2.set_ylabel('RSI')
        ax2.set_xlabel('Date')
        ax2.set_title('Relative Strength Index (RSI)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_ylim(0, 100) # RSI is always between 0 and 100

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Only show the plot if it was created internally (i.e., ax was None)
        if ax is None:
            plt.show()