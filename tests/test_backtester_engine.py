import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the class under test
from src.backtester.engine import BacktestEngine
from src.strategies.base_strategy import BaseStrategy
from src.backtester.portfolio_manager import PortfolioManager # Used for mocking
from src.backtester.risk_manager import RiskManager # Used for mocking

# --- Mock Classes for Testing ---

class MockStrategy(BaseStrategy):
    """
    A mock strategy for testing the BacktestEngine.
    It returns pre-defined signals.
    """
    def __init__(self, signals_df: pd.DataFrame):
        self.signals_df = signals_df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        For testing, this method simply returns the pre-defined signals DataFrame.
        In a real strategy, 'data' would be used to compute these signals.
        """
        return self.signals_df

    def get_name(self) -> str:
        """Returns the name of the mock strategy."""
        return "MockStrategy"

# --- Helper Functions for Test Data ---

def create_sample_data(num_days=10, start_price=100) -> pd.DataFrame:
    """
    Creates a sample historical data DataFrame for testing.
    """
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
    data = {
        'Open': np.linspace(start_price, start_price + num_days - 1, num_days),
        'High': np.linspace(start_price + 2, start_price + num_days + 1, num_days),
        'Low': np.linspace(start_price - 2, start_price + num_days - 3, num_days),
        'Close': np.linspace(start_price + 1, start_price + num_days, num_days),
        'Volume': np.random.randint(100000, 500000, num_days)
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

def create_sample_signals(data_df: pd.DataFrame, signals_list: list) -> pd.DataFrame:
    """
    Creates a sample signals DataFrame based on a list of (date_index, signal_type, price) tuples.

    Args:
        data_df (pd.DataFrame): The historical data DataFrame to get dates from.
        signals_list (list): A list of tuples, e.g., [(0, 'BUY', 101.0), (2, 'SELL', 103.0)].
                             date_index refers to the integer index of the date in data_df.

    Returns:
        pd.DataFrame: A DataFrame with 'signal' and 'price' columns, indexed by date.
    """
    signals_df = pd.DataFrame(index=data_df.index, columns=['signal', 'price'])
    for idx, signal_type, price in signals_list:
        if idx < len(data_df):
            signals_df.loc[data_df.index[idx], 'signal'] = signal_type
            signals_df.loc[data_df.index[idx], 'price'] = price
    return signals_df.dropna(subset=['signal']) # Only keep rows with actual signals

# --- Pytest Fixtures ---

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Provides a default sample historical data DataFrame."""
    return create_sample_data()

@pytest.fixture
def mock_strategy_no_signals(sample_data) -> MockStrategy:
    """Provides a mock strategy that generates no signals."""
    signals_df = create_sample_signals(sample_data, [])
    return MockStrategy(signals_df)

@pytest.fixture
def mock_strategy_buy_sell(sample_data) -> MockStrategy:
    """
    Provides a mock strategy with a simple buy on day 0 and sell on day 2.
    """
    signals = [
        (0, 'BUY', sample_data['Close'].iloc[0]),
        (2, 'SELL', sample_data['Close'].iloc[2])
    ]
    signals_df = create_sample_signals(sample_data, signals)
    return MockStrategy(signals_df)

@pytest.fixture
def mock_strategy_multiple_buys(sample_data) -> MockStrategy:
    """
    Provides a mock strategy with multiple buy signals followed by a sell.
    """
    signals = [
        (0, 'BUY', sample_data['Close'].iloc[0]),
        (1, 'BUY', sample_data['Close'].iloc[1]),
        (3, 'SELL', sample_data['Close'].iloc[3])
    ]
    signals_df = create_sample_signals(sample_data, signals)
    return MockStrategy(signals_df)

# --- Tests for BacktestEngine ---

class TestBacktestEngine:
    """
    Unit tests for the BacktestEngine class.
    Mocks PortfolioManager and RiskManager to isolate engine logic.
    """
    INITIAL_CAPITAL = 100000.0
    COMMISSION_RATE = 0.001
    ASSET_SYMBOL = 'TEST' # Default symbol used internally by engine

    @patch('src.backtester.portfolio_manager.PortfolioManager')
    @patch('src.backtester.risk_manager.RiskManager')
    def test_engine_initialization(self, MockRiskManager, MockPortfolioManager, sample_data, mock_strategy_no_signals):
        """
        Tests if the BacktestEngine initializes correctly with provided parameters.
        """
        engine = BacktestEngine(
            historical_data=sample_data,
            strategy=mock_strategy_no_signals,
            initial_capital=self.INITIAL_CAPITAL,
            commission_rate=self.COMMISSION_RATE
        )
        assert engine.historical_data.equals(sample_data)
        assert engine.strategy == mock_strategy_no_signals
        assert engine.initial_capital == self.INITIAL_CAPITAL
        assert engine.commission_rate == self.COMMISSION_RATE
        assert engine.portfolio_history == []
        assert engine.trade_log == []

    @patch('src.backtester.portfolio_manager.PortfolioManager')
    @patch('src.backtester.risk_manager.RiskManager')
    def test_run_backtest_no_signals(self, MockRiskManager, MockPortfolioManager, sample_data, mock_strategy_no_signals):
        """
        Tests the backtest run when the strategy generates no signals.
        Portfolio value should remain constant, and no trades should occur.
        """
        # Configure mock PortfolioManager to return constant values
        mock_pm_instance = MockPortfolioManager.return_value
        mock_pm_instance.get_current_portfolio_value.return_value = self.INITIAL_CAPITAL
        mock_pm_instance.get_cash_balance.return_value = self.INITIAL_CAPITAL
        mock_pm_instance.get_current_positions.return_value = {}

        engine = BacktestEngine(
            historical_data=sample_data,
            strategy=mock_strategy_no_signals,
            initial_capital=self.INITIAL_CAPITAL,
            commission_rate=self.COMMISSION_RATE
        )
        portfolio_history, trade_log = engine.run_backtest()

        assert len(portfolio_history) == len(sample_data)
        # All portfolio history entries should reflect the initial capital
        assert all(entry['total_value'] == pytest.approx(self.INITIAL_CAPITAL) for entry in portfolio_history)
        assert len(trade_log) == 0 # No trades should be logged

        # Verify PortfolioManager methods were called correctly
        mock_pm_instance.initialize_portfolio.assert_called_once_with(self.INITIAL_CAPITAL)
        mock_pm_instance.buy_asset.assert_not_called()
        mock_pm_instance.sell_asset.assert_not_called()

    @patch('src.backtester.portfolio_manager.PortfolioManager')
    @patch('src.backtester.risk_manager.RiskManager')
    def test_run_backtest_basic_buy_sell(self, MockRiskManager, MockPortfolioManager, sample_data, mock_strategy_buy_sell):
        """
        Tests a basic scenario with one buy and one sell signal.
        Verifies trade logging and calls to PortfolioManager/RiskManager.
        """
        mock_pm_instance = MockPortfolioManager.return_value
        mock_rm_instance = MockRiskManager.return_value

        # Mock position sizing: always buy/sell 10 units
        mock_rm_instance.calculate_position_size.return_value = 10

        # Simulate successful buys/sells by returning the quantity traded
        mock_pm_instance.buy_asset.return_value = 10
        mock_pm_instance.sell_asset.return_value = 10

        # Simulate portfolio state changes for history recording (simplified)
        initial_cash = self.INITIAL_CAPITAL
        buy_price_day0 = sample_data['Close'].iloc[0] # e.g., 101.0
        sell_price_day2 = sample_data['Close'].iloc[2] # e.g., 103.0

        # Define expected portfolio values at end of each day
        # This is simplified for testing engine's recording, not PM's calculation
        expected_values = [
            initial_cash, # Day 0 before trade
            initial_cash - (10 * buy_price_day0 * (1 + self.COMMISSION_RATE)), # Day 0 after buy
            initial_cash - (10 * buy_price_day0 * (1 + self.COMMISSION_RATE)) + (10 * (sample_data['Close'].iloc[1] - buy_price_day0)), # Day 1 (value changes with price)
            initial_cash - (10 * buy_price_day0 * (1 + self.COMMISSION_RATE)) + (10 * (sample_data['Close'].iloc[2] - buy_price_day0)), # Day 2 before sell
            initial_cash - (10 * buy_price_day0 * (1 + self.COMMISSION_RATE)) + (10 * sell_price_day2 * (1 - self.COMMISSION_RATE)), # Day 2 after sell
        ]
        # Extend with final value for remaining days
        final_value = expected_values[-1]
        expected_values.extend([final_value] * (len(sample_data) - len(expected_values)))

        mock_pm_instance.get_current_portfolio_value.side_effect = expected_values
        mock_pm_instance.get_cash_balance.return_value = initial_cash # Simplified
        mock_pm_instance.get_current_positions.return_value = {} # Simplified

        engine = BacktestEngine(
            historical_data=sample_data,
            strategy=mock_strategy_buy_sell,
            initial_capital=self.INITIAL_CAPITAL,
            commission_rate=self.COMMISSION_RATE
        )
        portfolio_history, trade_log = engine.run_backtest()

        assert len(portfolio_history) == len(sample_data)
        assert len(trade_log) == 2 # One buy, one sell

        # Verify buy trade on Day 0
        buy_trade = trade_log[0]
        assert buy_trade['type'] == 'BUY'
        assert buy_trade['quantity'] == 10
        assert buy_trade['price'] == pytest.approx(buy_price_day0)
        assert buy_trade['commission'] == pytest.approx(10 * buy_price_day0 * self.COMMISSION_RATE)
        assert buy_trade['date'] == sample_data.index[0]

        # Verify sell trade on Day 2
        sell_trade = trade_log[1]
        assert sell_trade['type'] == 'SELL'
        assert sell_trade['quantity'] == 10
        assert sell_trade['price'] == pytest.approx(sell_price_day2)
        assert sell_trade['commission'] == pytest.approx(10 * sell_price_day2 * self.COMMISSION_RATE)
        assert sell_trade['date'] == sample_data.index[2]

        # Verify PortfolioManager methods were called correctly
        mock_pm_instance.initialize_portfolio.assert_called_once_with(self.INITIAL_CAPITAL)
        mock_pm_instance.buy_asset.assert_called_once_with(
            symbol=self.ASSET_SYMBOL,
            quantity=10,
            price=buy_price_day0,
            commission_rate=self.COMMISSION_RATE,
            trade_date=sample_data.index[0]
        )
        mock_pm_instance.sell_asset.assert_called_once_with(
            symbol=self.ASSET_SYMBOL,
            quantity=10,
            price=sell_price_day2,
            commission_rate=self.COMMISSION_RATE,
            trade_date=sample_data.index[2]
        )
        assert mock_rm_instance.calculate_position_size.call_count == 2 # Called for buy and sell

        # Verify portfolio history values (structure and dates)
        for i, entry in enumerate(portfolio_history):
            assert entry['date'] == sample_data.index[i]
            assert 'total_value' in entry
            assert 'cash' in entry
            assert 'positions' in entry
            assert 'daily_return' in entry

    @patch('src.backtester.portfolio_manager.PortfolioManager')
    @patch('src.backtester.risk_manager.RiskManager')
    def test_run_backtest_insufficient_funds(self, MockRiskManager, MockPortfolioManager, sample_data, mock_strategy_buy_sell):
        """
        Tests how the engine handles a buy signal when there are insufficient funds.
        Assumes PortfolioManager returns 0 units bought in such a scenario.
        """
        low_capital = 500.0 # Set initial capital very low
        mock_pm_instance = MockPortfolioManager.return_value
        mock_rm_instance = MockRiskManager.return_value

        # Mock position sizing: always try to buy 10 units (which will be too much)
        mock_rm_instance.calculate_position_size.return_value = 10

        # Simulate PortfolioManager's buy_asset returning 0 units if funds are insufficient
        # This mocks the behavior of PortfolioManager, not the engine's decision
        def mock_buy_asset_side_effect(symbol, quantity, price, commission_rate, trade_date):
            cost = quantity * price * (1 + commission_rate)
            if cost > mock_pm_instance.get_cash_balance.return_value:
                return 0 # Simulate no units bought due to insufficient funds
            else:
                return quantity # Should not be reached in this test

        mock_pm_instance.buy_asset.side_effect = mock_buy_asset_side_effect
        mock_pm_instance.get_cash_balance.return_value = low_capital # Initial cash
        mock_pm_instance.get_current_portfolio_value.return_value = low_capital
        mock_pm_instance.get_current_positions.return_value = {}

        engine = BacktestEngine(
            historical_data=sample_data,
            strategy=mock_strategy_buy_sell, # Strategy tries to buy 10 units on Day 0
            initial_capital=low_capital,
            commission_rate=self.COMMISSION_RATE
        )
        portfolio_history, trade_log = engine.run_backtest()

        # The first buy (10 units at ~101.0) would cost ~1010, which is > 500.
        # So, the buy_asset call should result in 0 units bought, and no trade recorded.
        assert len(trade_log) == 0

        # Verify buy_asset was called (engine attempted the trade)
        mock_pm_instance.buy_asset.assert_called_once_with(
            symbol=self.ASSET_SYMBOL,
            quantity=10, # Engine requested 10 units
            price=pytest.approx(sample_data['Close'].iloc[0]),
            commission_rate=self.COMMISSION_RATE,
            trade_date=sample_data.index[0]
        )
        # If no buy occurred, no sell should occur either
        mock_pm_instance.sell_asset.assert_not_called()

        assert all(entry['total_value'] == pytest.approx(low_capital) for entry in portfolio_history)

    @patch('src.backtester.portfolio_manager.PortfolioManager')
    @patch('src.backtester.risk_manager.RiskManager')
    def test_run_backtest_multiple_buys_and_single_sell(self, MockRiskManager, MockPortfolioManager, sample_data, mock_strategy_multiple_buys):
        """
        Tests a scenario with multiple buy signals followed by a single sell signal.
        Verifies correct quantities are passed to PortfolioManager.
        """
        mock_pm_instance = MockPortfolioManager.return_value
        mock_rm_instance = MockRiskManager.return_value

        # Mock position sizing: Buy 10, then Buy 5, then Sell all (15)
        mock_rm_instance.calculate_position_size.side_effect = [10, 5, 15]

        # Simulate successful buys/sells by returning the quantity
        mock_pm_instance.buy_asset.side_effect = lambda symbol, quantity, price, commission_rate, trade_date: quantity
        mock_pm_instance.sell_asset.side_effect = lambda symbol, quantity, price, commission_rate, trade_date: quantity

        # Simplified PM state for this test, focusing on calls
        mock_pm_instance.get_current_portfolio_value.return_value = self.INITIAL_CAPITAL
        mock_pm_instance.get_cash_balance.return_value = self.INITIAL_CAPITAL
        mock_pm_instance.get_current_positions.return_value = {self.ASSET_SYMBOL: 0} # Start with 0 position

        engine = BacktestEngine(
            historical_data=sample_data,
            strategy=mock_strategy_multiple_buys,
            initial_capital=self.INITIAL_CAPITAL,
            commission_rate=self.COMMISSION_RATE
        )
        portfolio_history, trade_log = engine.run_backtest()

        assert len(trade_log) == 3 # Two buys, one sell

        # Verify buy on Day 0
        buy_trade_0 = trade_log[0]
        assert buy_trade_0['type'] == 'BUY'
        assert buy_trade_0['quantity'] == 10
        assert buy_trade_0['price'] == pytest.approx(sample_data['Close'].iloc[0])

        # Verify buy on Day 1
        buy_trade_1 = trade_log[1]
        assert buy_trade_1['type'] == 'BUY'
        assert buy_trade_1['quantity'] == 5
        assert buy_trade_1['price'] == pytest.approx(sample_data['Close'].iloc[1])

        # Verify sell on Day 3
        sell_trade_3 = trade_log[2]
        assert sell_trade_3['type'] == 'SELL'
        assert sell_trade_3['quantity'] == 15 # Should sell total quantity held
        assert sell_trade_3['price'] == pytest.approx(sample_data['Close'].iloc[3])

        # Verify calls to PortfolioManager
        mock_pm_instance.buy_asset.assert_any_call(
            symbol=self.ASSET_SYMBOL, quantity=10, price=pytest.approx(sample_data['Close'].iloc[0]),
            commission_rate=self.COMMISSION_RATE, trade_date=sample_data.index[0]
        )
        mock_pm_instance.buy_asset.assert_any_call(
            symbol=self.ASSET_SYMBOL, quantity=5, price=pytest.approx(sample_data['Close'].iloc[1]),
            commission_rate=self.COMMISSION_RATE, trade_date=sample_data.index[1]
        )
        mock_pm_instance.sell_asset.assert_called_once_with(
            symbol=self.ASSET_SYMBOL, quantity=15, price=pytest.approx(sample_data['Close'].iloc[3]),
            commission_rate=self.COMMISSION_RATE, trade_date=sample_data.index[3]
        )
        assert mock_pm_instance.buy_asset.call_count == 2
        assert mock_pm_instance.sell_asset.call_count == 1
        assert mock_rm_instance.calculate_position_size.call_count == 3

    @patch('src.backtester.portfolio_manager.PortfolioManager')
    @patch('src.backtester.risk_manager.RiskManager')
    def test_run_backtest_sell_without_position(self, MockRiskManager, MockPortfolioManager, sample_data):
        """
        Tests the engine's behavior when a sell signal is generated but no position is held.
        Assumes PortfolioManager returns 0 units sold in such a scenario.
        """
        signals = [
            (0, 'SELL', sample_data['Close'].iloc[0]) # Try to sell on Day 0
        ]
        mock_strategy = MockStrategy(create_sample_signals(sample_data, signals))

        mock_pm_instance = MockPortfolioManager.return_value
        mock_rm_instance = MockRiskManager.return_value

        # Mock PM to indicate no positions held
        mock_pm_instance.get_current_positions.return_value = {}
        mock_pm_instance.get_cash_balance.return_value = self.INITIAL_CAPITAL
        mock_pm_instance.get_current_portfolio_value.return_value = self.INITIAL_CAPITAL

        # Simulate sell_asset returning 0 if no position is held
        mock_pm_instance.sell_asset.return_value = 0

        engine = BacktestEngine(
            historical_data=sample_data,
            strategy=mock_strategy,
            initial_capital=self.INITIAL_CAPITAL,
            commission_rate=self.COMMISSION_RATE
        )
        portfolio_history, trade_log = engine.run_backtest()

        assert len(trade_log) == 0 # No trades should be recorded
        mock_pm_instance.sell_asset.assert_called_once() # Engine tried to sell
        mock_pm_instance.buy_asset.assert_not_called()

        assert all(entry['total_value'] == pytest.approx(self.INITIAL_CAPITAL) for entry in portfolio_history)

    @patch('src.backtester.portfolio_manager.PortfolioManager')
    @patch('src.backtester.risk_manager.RiskManager')
    def test_engine_handles_empty_historical_data(self, MockRiskManager, MockPortfolioManager):
        """
        Tests that the engine correctly handles an empty historical data DataFrame.
        No backtest should run, and history/logs should be empty.
        """
        empty_data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'], index=pd.DatetimeIndex([]))
        mock_strategy = MockStrategy(pd.DataFrame(columns=['signal', 'price']))

        engine = BacktestEngine(
            historical_data=empty_data,
            strategy=mock_strategy,
            initial_capital=self.INITIAL_CAPITAL,
            commission_rate=self.COMMISSION_RATE
        )
        portfolio_history, trade_log = engine.run_backtest()

        assert len(portfolio_history) == 0
        assert len(trade_log) == 0
        # PortfolioManager should not be initialized or interacted with if no data
        MockPortfolioManager.return_value.initialize_portfolio.assert_not_called()
        MockPortfolioManager.return_value.get_current_portfolio_value.assert_not_called()
        MockRiskManager.return_value.calculate_position_size.assert_not_called()

    @patch('src.backtester.portfolio_manager.PortfolioManager')
    @patch('src.backtester.risk_manager.RiskManager')
    def test_portfolio_history_and_daily_return_calculation(self, MockRiskManager, MockPortfolioManager, sample_data):
        """
        Tests the accuracy of portfolio history recording, especially daily returns.
        Mocks portfolio values explicitly to verify engine's calculations.
        """
        mock_pm_instance = MockPortfolioManager.return_value
        mock_rm_instance = MockRiskManager.return_value

        # Define a sequence of portfolio values for the mock PM to return
        # This allows us to precisely test the daily return calculation
        mock_portfolio_values = [
            self.INITIAL_CAPITAL, # Day 0
            self.INITIAL_CAPITAL * 1.01, # Day 1: +1%
            self.INITIAL_CAPITAL * 1.01 * 0.99, # Day 2: -1%
            self.INITIAL_CAPITAL * 1.01 * 0.99 * 1.05, # Day 3: +5%
            self.INITIAL_CAPITAL * 1.01 * 0.99 * 1.05, # Day 4: no change
            self.INITIAL_CAPITAL * 1.01 * 0.99 * 1.05 * 0.98, # Day 5: -2%
            self.INITIAL_CAPITAL * 1.01 * 0.99 * 1.05 * 0.98, # Day 6
            self.INITIAL_CAPITAL * 1.01 * 0.99 * 1.05 * 0.98, # Day 7
            self.INITIAL_CAPITAL * 1.01 * 0.99 * 1.05 * 0.98, # Day 8
            self.INITIAL_CAPITAL * 1.01 * 0.99 * 1.05 * 0.98, # Day 9
        ]
        mock_pm_instance.get_current_portfolio_value.side_effect = mock_portfolio_values
        mock_pm_instance.get_cash_balance.return_value = self.INITIAL_CAPITAL # Simplified
        mock_pm_instance.get_current_positions.return_value = {} # Simplified

        # No signals generated for this test, so no trades.
        mock_strategy = MockStrategy(create_sample_signals(sample_data, []))

        engine = BacktestEngine(
            historical_data=sample_data,
            strategy=mock_strategy,
            initial_capital=self.INITIAL_CAPITAL,
            commission_rate=self.COMMISSION_RATE
        )
        portfolio_history, _ = engine.run_backtest()

        assert len(portfolio_history) == len(sample_data)

        # Verify initial state (Day 0)
        assert portfolio_history[0]['date'] == sample_data.index[0]
        assert portfolio_history[0]['total_value'] == pytest.approx(mock_portfolio_values[0])
        assert portfolio_history[0]['daily_return'] == pytest.approx(0.0) # First day has no previous return

        # Verify daily returns for subsequent days
        for i in range(1, len(portfolio_history)):
            prev_value = portfolio_history[i-1]['total_value']
            curr_value = portfolio_history[i]['total_value']
            expected_daily_return = (curr_value / prev_value) - 1 if prev_value != 0 else 0.0

            assert portfolio_history[i]['date'] == sample_data.index[i]
            assert portfolio_history[i]['total_value'] == pytest.approx(mock_portfolio_values[i])
            assert portfolio_history[i]['daily_return'] == pytest.approx(expected_daily_return)

        mock_pm_instance.initialize_portfolio.assert_called_once_with(self.INITIAL_CAPITAL)
        mock_pm_instance.buy_asset.assert_not_called()
        mock_pm_instance.sell_asset.assert_not_called()