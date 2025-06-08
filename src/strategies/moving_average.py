import pandas as pd
import pandas_ta as ta
import logging
from src.strategies.base_strategy import BaseStrategy

# Configure logging for this module
logger = logging.getLogger(__name__)

class MovingAverageStrategy(BaseStrategy):
    """
    Implements a simple Moving Average Crossover trading strategy.

    This strategy generates a buy signal when a shorter-period moving average
    crosses above a longer-period moving average, and a sell signal when
    the shorter-period moving average crosses below the longer-period one.

    Parameters:
    - short_window (int): The period for the short-term moving average.
    - long_window (int): The period for the long-term moving average.
    - ma_type (str): Type of moving average to use ('SMA' for Simple, 'EMA' for Exponential).
                     Defaults to 'SMA'.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50, ma_type: str = 'SMA'):
        """
        Initializes the MovingAverageStrategy with specified parameters.
        """
        if short_window >= long_window:
            raise ValueError("Short window must be less than long window.")
        if ma_type.upper() not in ['SMA', 'EMA']:
            raise ValueError("ma_type must be 'SMA' or 'EMA'.")

        super().__init__(name=f"MA_Cross_{short_window}_{long_window}_{ma_type.upper()}")
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type.upper()
        logger.info(f"Initialized {self.name} with short_window={self.short_window}, "
                    f"long_window={self.long_window}, ma_type={self.ma_type}")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals (buy/sell/hold) based on the moving average crossover strategy.

        Args:
            data (pd.DataFrame): Historical stock data with at least a 'Close' column.

        Returns:
            pd.DataFrame: A DataFrame with an 'signal' column (1 for buy, -1 for sell, 0 for hold),
                          indexed by date.
        """
        if 'Close' not in data.columns:
            logger.error("Input data must contain a 'Close' column for MA strategy.")
            raise ValueError("Input data must contain a 'Close' column.")

        if len(data) < self.long_window:
            logger.warning(f"Not enough data points ({len(data)}) for MA strategy with long window {self.long_window}. "
                           "Returning empty signals.")
            return pd.DataFrame(index=data.index, data={'signal': 0})

        logger.debug(f"Generating signals for {self.name} on data of shape {data.shape}")

        # Calculate moving averages using pandas_ta
        if self.ma_type == 'SMA':
            data[f'SMA_{self.short_window}'] = ta.sma(data['Close'], length=self.short_window)
            data[f'SMA_{self.long_window}'] = ta.sma(data['Close'], length=self.long_window)
            short_ma = data[f'SMA_{self.short_window}']
            long_ma = data[f'SMA_{self.long_window}']
        elif self.ma_type == 'EMA':
            data[f'EMA_{self.short_window}'] = ta.ema(data['Close'], length=self.short_window)
            data[f'EMA_{self.long_window}'] = ta.ema(data['Close'], length=self.long_window)
            short_ma = data[f'EMA_{self.short_window}']
            long_ma = data[f'EMA_{self.long_window}']
        else:
            # This case should ideally be caught in __init__
            raise ValueError(f"Unsupported MA type: {self.ma_type}")

        # Initialize signals column
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0  # Default to hold

        # Generate buy signals
        # Buy when short MA crosses above long MA
        buy_condition = (short_ma.shift(1) < long_ma.shift(1)) & (short_ma > long_ma)
        signals.loc[buy_condition, 'signal'] = 1

        # Generate sell signals
        # Sell when short MA crosses below long MA
        sell_condition = (short_ma.shift(1) > long_ma.shift(1)) & (short_ma < long_ma)
        signals.loc[sell_condition, 'signal'] = -1

        # Drop NA values that result from MA calculation (at the beginning of the series)
        signals = signals.dropna()
        logger.debug(f"Generated {signals['signal'].value_counts().get(1, 0)} buy signals, "
                     f"{signals['signal'].value_counts().get(-1, 0)} sell signals.")

        return signals

    def get_strategy_params(self) -> dict:
        """
        Returns a dictionary of the strategy's parameters.
        """
        return {
            'name': self.name,
            'short_window': self.short_window,
            'long_window': self.long_window,
            'ma_type': self.ma_type
        }

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Create dummy data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    close_prices = pd.Series([100 + i + (i % 10) * 2 - (i % 5) * 5 for i in range(200)], index=dates)
    dummy_data = pd.DataFrame({'Close': close_prices})

    print("--- Dummy Data Head ---")
    print(dummy_data.head())
    print("\n--- Dummy Data Tail ---")
    print(dummy_data.tail())

    # Initialize the strategy
    ma_strategy = MovingAverageStrategy(short_window=10, long_window=30, ma_type='EMA')

    # Generate signals
    signals_df = ma_strategy.generate_signals(dummy_data.copy()) # Use .copy() to avoid modifying original df

    print(f"\n--- Signals Generated by {ma_strategy.name} ---")
    print(signals_df[signals_df['signal'] != 0].head(10))
    print(signals_df[signals_df['signal'] != 0].tail(10))

    buy_count = signals_df[signals_df['signal'] == 1].shape[0]
    sell_count = signals_df[signals_df['signal'] == -1].shape[0]
    hold_count = signals_df[signals_df['signal'] == 0].shape[0]

    print(f"\nTotal Buy Signals: {buy_count}")
    print(f"Total Sell Signals: {sell_count}")
    print(f"Total Hold Signals: {hold_count}")
    print(f"Total Signals: {signals_df.shape[0]}")

    # Test with insufficient data
    print("\n--- Testing with insufficient data ---")
    small_data = dummy_data.head(20)
    try:
        ma_strategy_small = MovingAverageStrategy(short_window=10, long_window=30)
        signals_small = ma_strategy_small.generate_signals(small_data.copy())
        print("Signals generated for small data (expected to be mostly 0s):")
        print(signals_small.value_counts())
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Test parameter retrieval
    print("\n--- Strategy Parameters ---")
    print(ma_strategy.get_strategy_params())