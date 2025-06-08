import pytest
import pandas as pd
import numpy as np
import pandas_ta as ta

from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.base_strategy import BaseStrategy # For type checking if needed

# Fixture for a generic DataFrame to be used in tests
@pytest.fixture
def sample_dataframe():
    """
    Provides a generic pandas DataFrame simulating historical stock data.
    The data has a mix of trends and noise to allow for various test scenarios.
    """
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))
    # Create a base price series with some trends
    base_prices = np.concatenate([
        np.linspace(100, 120, 30),  # Upward trend
        np.linspace(120, 90, 30),   # Downward trend
        np.linspace(90, 110, 40)    # Recovery/mild upward trend
    ])
    # Add some random noise to make it more realistic
    close_prices = base_prices + np.random.normal(0, 0.5, 100)
    # Ensure prices are always positive
    close_prices[close_prices < 1] = 1

    df = pd.DataFrame({
        'Open': close_prices * 0.99,
        'High': close_prices * 1.01,
        'Low': close_prices * 0.98,
        'Close': close_prices,
        'Volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)
    return df

class TestMovingAverageStrategy:
    """
    Unit tests for the MovingAverageStrategy.
    """

    def test_initialization(self):
        """
        Tests that the strategy initializes with the correct parameters.
        """
        strategy = MovingAverageStrategy(short_period=10, long_period=20)
        assert strategy.short_period == 10
        assert strategy.long_period == 20
        assert isinstance(strategy, BaseStrategy)

    def test_not_enough_data(self):
        """
        Tests that generate_signals returns all zeros when there isn't enough
        data to calculate the moving averages (i.e., data length < long_period).
        """
        # Data shorter than long_period (5)
        data = pd.DataFrame({'Close': [10, 11, 12, 13]},
                            index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=4)))
        strategy = MovingAverageStrategy(short_period=3, long_period=5)
        signals = strategy.generate_signals(data)

        # Expected signals should be all zeros as per BaseStrategy's fillna(0)
        expected_signals = pd.Series([0, 0, 0, 0], index=data.index)
        pd.testing.assert_series_equal(signals, expected_signals, check_dtype=False, check_names=False)
        assert len(signals) == len(data)

    def test_generate_signals_crossover_buy_and_sell(self):
        """
        Tests that the strategy correctly generates buy and sell signals
        based on short and long moving average crossovers.
        """
        # Craft data for clear buy and sell signals
        # Short MA (3), Long MA (5)
        # Data designed to create a buy signal, then a sell signal.
        dates_ma = pd.to_datetime(pd.date_range(start='2023-01-01', periods=15))
        close_prices_ma = [
            10, 9, 8, 7, 6,  # Initial decline to set up short_ma < long_ma
            15, 16, 17, 18, 19, # Sharp rise to create short_ma > long_ma (BUY)
            10, 9, 8, 7, 6   # Sharp decline to create short_ma < long_ma (SELL)
        ]
        data_ma = pd.DataFrame({'Close': close_prices_ma}, index=dates_ma)

        strategy_ma = MovingAverageStrategy(short_period=3, long_period=5)
        signals_ma = strategy_ma.generate_signals(data_ma)

        # Manual calculation verification (using pandas_ta for consistency)
        # df_check = data_ma.copy()
        # df_check['SMA_3'] = ta.sma(df_check['Close'], length=3)
        # df_check['SMA_5'] = ta.sma(df_check['Close'], length=5)
        # df_check['prev_SMA_3'] = df_check['SMA_3'].shift(1)
        # df_check['prev_SMA_5'] = df_check['SMA_5'].shift(1)
        # df_check['buy_cond'] = (df_check['prev_SMA_3'] < df_check['prev_SMA_5']) & (df_check['SMA_3'] > df_check['SMA_5'])
        # df_check['sell_cond'] = (df_check['prev_SMA_3'] > df_check['prev_SMA_5']) & (df_check['SMA_3'] < df_check['SMA_5'])
        # print(df_check[['Close', 'SMA_3', 'SMA_5', 'buy_cond', 'sell_cond']])

        expected_signals_ma = pd.Series([
            0, 0, 0, 0, 0, # Not enough data for MA_5, then no crossover condition met yet
            1,             # Buy signal at index 5 (2023-01-06)
            0, 0, 0, 0,    # No crossover
            -1,            # Sell signal at index 10 (2023-01-11)
            0, 0, 0, 0     # No more crossovers
        ], index=dates_ma)

        pd.testing.assert_series_equal(signals_ma, expected_signals_ma, check_dtype=False, check_names=False)

    def test_generate_signals_no_crossover(self, sample_dataframe):
        """
        Tests that no signals are generated when there are no MA crossovers.
        Uses a consistently rising price trend.
        """
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=30))
        close_prices = np.linspace(100, 200, 30) # Consistent upward trend
        data = pd.DataFrame({'Close': close_prices}, index=dates)

        strategy = MovingAverageStrategy(short_period=5, long_period=10)
        signals = strategy.generate_signals(data)

        # After initial NaN fill, all signals should be 0 as no crossovers occur
        expected_signals = pd.Series(0, index=dates)
        pd.testing.assert_series_equal(signals, expected_signals, check_dtype=False, check_names=False)

    def test_signals_are_integers(self, sample_dataframe):
        """
        Ensures that the generated signals are of integer type and only contain -1, 0, or 1.
        """
        strategy = MovingAverageStrategy(short_period=10, long_period=20)
        signals = strategy.generate_signals(sample_dataframe)
        assert signals.dtype == 'int64'
        assert all(x in [-1, 0, 1] for x in signals.unique())


class TestRSIStrategy:
    """
    Unit tests for the RSIStrategy.
    """

    def test_initialization(self):
        """
        Tests that the strategy initializes with the correct parameters.
        """
        strategy = RSIStrategy(rsi_period=14, overbought_threshold=70, oversold_threshold=30)
        assert strategy.rsi_period == 14
        assert strategy.overbought_threshold == 70
        assert strategy.oversold_threshold == 30
        assert isinstance(strategy, BaseStrategy)

    def test_not_enough_data(self):
        """
        Tests that generate_signals returns all zeros when there isn't enough
        data to calculate RSI (i.e., data length < rsi_period + 1).
        """
        # Data shorter than rsi_period (14) + 1
        data = pd.DataFrame({'Close': [10, 11, 12, 13, 14]},
                            index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=5)))
        strategy = RSIStrategy(rsi_period=14, overbought_threshold=70, oversold_threshold=30)
        signals = strategy.generate_signals(data)

        # Expected signals should be all zeros as per BaseStrategy's fillna(0)
        expected_signals = pd.Series([0, 0, 0, 0, 0], index=data.index)
        pd.testing.assert_series_equal(signals, expected_signals, check_dtype=False, check_names=False)
        assert len(signals) == len(data)

    def test_generate_signals_rsi_thresholds(self):
        """
        Tests that the strategy correctly generates buy and sell signals
        based on RSI crossing oversold and overbought thresholds.
        """
        # Craft data for clear RSI buy and sell signals
        # Using a smaller RSI period (e.g., 6) for easier manual verification.
        # Oversold = 30, Overbought = 70
        dates_rsi = pd.to_datetime(pd.date_range(start='2023-01-01', periods=20))
        close_prices_rsi = [
            100, 100, 100, 100, 100, 100, # Initial flat values for RSI calculation to stabilize around 50
            90, 80, 70, 60, 50, # Drop to oversold (RSI will go below 30)
            60, 70, 80, 90, 100, 110, 120, 130, 140 # Rise to overbought (RSI will go above 70)
        ]
        data_rsi = pd.DataFrame({'Close': close_prices_rsi}, index=dates_rsi)

        strategy_rsi = RSIStrategy(rsi_period=6, overbought_threshold=70, oversold_threshold=30)
        signals_rsi = strategy_rsi.generate_signals(data_rsi)

        # Manual calculation verification (using pandas_ta for consistency)
        # df_check = data_rsi.copy()
        # df_check['RSI'] = ta.rsi(df_check['Close'], length=6)
        # df_check['prev_RSI'] = df_check['RSI'].shift(1)
        # oversold_threshold = 30
        # overbought_threshold = 70
        # df_check['buy_cond'] = (df_check['prev_RSI'] > oversold_threshold) & (df_check['RSI'] <= oversold_threshold)
        # df_check['sell_cond'] = (df_check['prev_RSI'] < overbought_threshold) & (df_check['RSI'] >= overbought_threshold)
        # print(df_check[['Close', 'RSI', 'buy_cond', 'sell_cond']])

        expected_signals_rsi = pd.Series([
            0, 0, 0, 0, 0, 0, # Initial NaNs for RSI
            1, # Buy signal at index 6 (RSI crosses <= 30)
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # No signal
            -1, # Sell signal at index 17 (RSI crosses >= 70)
            0, 0 # No more signals
        ], index=dates_rsi)

        pd.testing.assert_series_equal(signals_rsi, expected_signals_rsi, check_dtype=False, check_names=False)

    def test_generate_signals_no_signals(self, sample_dataframe):
        """
        Tests that no signals are generated when RSI stays within the thresholds.
        Uses a dataset where prices are relatively stable, keeping RSI around 50.
        """
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=30))
        close_prices = np.linspace(100, 105, 30) # Very slight upward trend, keeps RSI stable
        data = pd.DataFrame({'Close': close_prices}, index=dates)

        strategy = RSIStrategy(rsi_period=14, overbought_threshold=70, oversold_threshold=30)
        signals = strategy.generate_signals(data)

        # After initial NaN fill, all signals should be 0 as RSI stays within bounds
        expected_signals = pd.Series(0, index=dates)
        pd.testing.assert_series_equal(signals, expected_signals, check_dtype=False, check_names=False)

    def test_signals_are_integers(self, sample_dataframe):
        """
        Ensures that the generated signals are of integer type and only contain -1, 0, or 1.
        """
        strategy = RSIStrategy(rsi_period=14, overbought_threshold=70, oversold_threshold=30)
        signals = strategy.generate_signals(sample_dataframe)
        assert signals.dtype == 'int64'
        assert all(x in [-1, 0, 1] for x in signals.unique())