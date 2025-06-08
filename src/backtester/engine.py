import pandas as pd
import numpy as np
import logging
from datetime import datetime

from src.strategies.base_strategy import BaseStrategy

# Configure logging for this module
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    The core simulation engine that iterates through historical data,
    executes trades based on strategy signals, and manages the backtest flow.
    """

    def __init__(self, data: pd.DataFrame, strategy: BaseStrategy, initial_cash: float,
                 commission_per_trade: float = 0.001, slippage_bps: int = 1, symbol: str = None):
        """
        Initializes the backtesting engine.

        Args:
            data (pd.DataFrame): Historical data (OHLCV + indicators) indexed by date.
                                 Must contain 'Close' price.
            strategy (BaseStrategy): An instance of a trading strategy.
            initial_cash (float): The starting capital for the backtest.
            commission_per_trade (float): Commission rate as a percentage of trade value (e.g., 0.001 for 0.1%).
            slippage_bps (int): Slippage in basis points (e.g., 1 for 0.01%).
                                 Applied to execution price: buy price increases, sell price decreases.
            symbol (str): The ticker symbol of the asset being traded. Required for position tracking.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Historical data must be a non-empty pandas DataFrame.")
        if 'Close' not in data.columns:
            raise ValueError("Data DataFrame must contain a 'Close' price column.")
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("Strategy must be an instance of BaseStrategy.")
        if initial_cash <= 0:
            raise ValueError("Initial cash must be positive.")
        if symbol is None:
            raise ValueError("Symbol must be provided for the asset being traded.")

        self.data = data
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps
        self.symbol = symbol

        # Portfolio state
        self._current_cash = initial_cash
        self._positions = {self.symbol: 0} # {symbol: quantity}
        self._portfolio_history = [] # List of dicts: {'date', 'cash', 'position_value', 'total_value'}
        self._transactions = []      # List of dicts: {'date', 'type', 'symbol', 'quantity', 'price', 'commission', 'net_amount', 'cash_balance_after'}

        logger.info(f"Backtest Engine initialized for {self.symbol} with initial cash: ${initial_cash:,.2f}")
        logger.info(f"Commission: {commission_per_trade*100:.2f}%, Slippage: {slippage_bps} bps")

    def _execute_trade(self, date: datetime, trade_type: str, price: float, quantity_to_trade: int = None):
        """
        Executes a trade (buy/sell) for the specified symbol.

        Args:
            date (datetime): The date of the trade.
            trade_type (str): 'buy' or 'sell'.
            price (float): The base price at which the trade is attempted.
            quantity_to_trade (int, optional): The exact quantity to trade. If None,
                                               engine decides (max possible for buy, all held for sell).
        """
        if trade_type not in ['buy', 'sell']:
            logger.error(f"Invalid trade type: {trade_type}")
            return

        executed_quantity = 0
        executed_price = price
        commission_amount = 0
        net_amount = 0
        transaction_status = "FAILED"

        # Apply slippage
        slippage_factor = self.slippage_bps / 10000.0
        if trade_type == 'buy':
            executed_price *= (1 + slippage_factor)
        elif trade_type == 'sell':
            executed_price *= (1 - slippage_factor)

        if trade_type == 'buy':
            # Determine quantity to buy
            if quantity_to_trade is None:
                # Buy with all available cash, considering commission and slippage
                # Calculate max shares that can be bought given current cash
                # total_cost_per_share = executed_price * (1 + self.commission_per_trade)
                # max_buyable_shares = int(self._current_cash / total_cost_per_share)
                # A more robust calculation considering commission on total cost:
                # C = current_cash, P = executed_price, Comm = commission_per_trade
                # Q = C / (P * (1 + Comm))
                max_buyable_shares = int(self._current_cash / (executed_price * (1 + self.commission_per_trade)))
                executed_quantity = max_buyable_shares
            else:
                executed_quantity = quantity_to_trade

            if executed_quantity <= 0:
                logger.debug(f"{date.strftime('%Y-%m-%d')}: Buy - No shares to buy or insufficient funds (cash: {self._current_cash:,.2f}, price: {executed_price:,.2f}).")
                return

            cost_before_commission = executed_quantity * executed_price
            commission_amount = cost_before_commission * self.commission_per_trade
            total_cost = cost_before_commission + commission_amount

            if total_cost <= self._current_cash:
                self._current_cash -= total_cost
                self._positions[self.symbol] += executed_quantity
                net_amount = -total_cost
                transaction_status = "EXECUTED"
                logger.info(f"{date.strftime('%Y-%m-%d')}: BUY {executed_quantity} shares of {self.symbol} @ ${executed_price:,.2f} (incl. slippage). Total Cost: ${total_cost:,.2f} (Comm: ${commission_amount:,.2f}).")
            else:
                logger.warning(f"{date.strftime('%Y-%m-%d')}: Buy - Insufficient cash to buy {executed_quantity} shares of {self.symbol}. Needed ${total_cost:,.2f}, have ${self._current_cash:,.2f}.")
                executed_quantity = 0 # No shares were bought if funds are insufficient
        
        elif trade_type == 'sell':
            # Determine quantity to sell
            shares_held = self._positions.get(self.symbol, 0)
            if quantity_to_trade is None:
                # Sell all held shares
                executed_quantity = shares_held
            else:
                executed_quantity = min(quantity_to_trade, shares_held)

            if executed_quantity <= 0:
                logger.debug(f"{date.strftime('%Y-%m-%d')}: Sell - No shares of {self.symbol} to sell.")
                return

            revenue_before_commission = executed_quantity * executed_price
            commission_amount = revenue_before_commission * self.commission_per_trade
            net_proceeds = revenue_before_commission - commission_amount

            self._current_cash += net_proceeds
            self._positions[self.symbol] -= executed_quantity
            net_amount = net_proceeds
            transaction_status = "EXECUTED"
            logger.info(f"{date.strftime('%Y-%m-%d')}: SELL {executed_quantity} shares of {self.symbol} @ ${executed_price:,.2f} (incl. slippage). Net Proceeds: ${net_proceeds:,.2f} (Comm: ${commission_amount:,.2f}).")

        if executed_quantity > 0:
            self._transactions.append({
                'date': date,
                'type': trade_type,
                'symbol': self.symbol,
                'quantity': executed_quantity,
                'price': price, # Original close price, before slippage
                'executed_price': executed_price, # Price after slippage
                'commission': commission_amount,
                'net_amount': net_amount, # Negative for buy, positive for sell
                'cash_balance_after': self._current_cash,
                'shares_held_after': self._positions[self.symbol],
                'status': transaction_status
            })

    def _record_portfolio_snapshot(self, date: datetime, current_price: float):
        """
        Records the current state of the portfolio.

        Args:
            date (datetime): The current date for the snapshot.
            current_price (float): The current market price of the asset.
        """
        shares_held = self._positions.get(self.symbol, 0)
        position_value = shares_held * current_price
        total_value = self._current_cash + position_value

        self._portfolio_history.append({
            'date': date,
            'cash': self._current_cash,
            'position_value': position_value,
            'total_value': total_value,
            'shares_held': shares_held
        })

    def run_backtest(self):
        """
        Runs the backtest simulation over the historical data.
        """
        logger.info(f"Starting backtest for {self.symbol}...")

        # Ensure data index is datetime for proper iteration and slicing
        if not isinstance(self.data.index, pd.DatetimeIndex):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except Exception as e:
                logger.error(f"Failed to convert data index to DatetimeIndex: {e}")
                return

        for i, (date, row) in enumerate(self.data.iterrows()):
            current_price = row['Close']

            # Generate signals for the current day
            # The strategy should be designed to take a slice of data up to the current day
            # This allows strategies to look back at historical indicators without look-ahead bias.
            current_data_slice = self.data.iloc[:i+1]
            signal = self.strategy.generate_signals(current_data_slice)

            logger.debug(f"{date.strftime('%Y-%m-%d')}: Signal: {signal}, Current Cash: ${self._current_cash:,.2f}, Shares: {self._positions[self.symbol]}")

            if signal == 'buy':
                # Attempt to buy. Engine decides quantity (max possible).
                self._execute_trade(date, 'buy', current_price)
            elif signal == 'sell':
                # Attempt to sell. Engine decides quantity (all held).
                self._execute_trade(date, 'sell', current_price)
            elif signal == 'hold':
                # No action, just log
                logger.debug(f"{date.strftime('%Y-%m-%d')}: HOLD")
            else:
                logger.warning(f"{date.strftime('%Y-%m-%d')}: Unknown signal '{signal}' from strategy.")

            # Record portfolio snapshot at the end of the day
            self._record_portfolio_snapshot(date, current_price)

        # Finalize: If there are open positions at the end, sell them to close the backtest.
        final_shares = self._positions.get(self.symbol, 0)
        if final_shares > 0:
            final_date = self.data.index[-1]
            final_price = self.data['Close'].iloc[-1]
            logger.info(f"Backtest ended. Liquidating {final_shares} shares of {self.symbol} at final price ${final_price:,.2f}.")
            self._execute_trade(final_date, 'sell', final_price, quantity_to_trade=final_shares)
            # Record one final snapshot after liquidation to reflect final cash position
            self._record_portfolio_snapshot(final_date, final_price)


        logger.info("Backtest completed.")

    def get_results(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns the backtest results: portfolio history and transaction log.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - portfolio_history_df (pd.DataFrame): DataFrame of daily portfolio snapshots.
                - transactions_df (pd.DataFrame): DataFrame of executed transactions.
        """
        portfolio_history_df = pd.DataFrame(self._portfolio_history)
        if not portfolio_history_df.empty:
            portfolio_history_df.set_index('date', inplace=True)

        transactions_df = pd.DataFrame(self._transactions)
        if not transactions_df.empty:
            transactions_df.set_index('date', inplace=True)

        return portfolio_history_df, transactions_df