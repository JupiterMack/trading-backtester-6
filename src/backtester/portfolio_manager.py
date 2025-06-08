import logging
from datetime import datetime
from typing import Dict, List, Any

# Configure logging for this module
logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    Manages the simulated portfolio's state, including cash, positions, and trade history.

    Attributes:
        initial_cash (float): The starting cash balance of the portfolio.
        cash (float): The current cash balance.
        positions (Dict[str, int]): A dictionary mapping ticker symbols to the
                                     quantity of shares held.
        trade_history (List[Dict[str, Any]]): A list of dictionaries, each representing a trade.
                                               Each trade record includes:
                                               - 'date': datetime object of the trade.
                                               - 'ticker': Symbol of the asset traded.
                                               - 'type': 'buy' or 'sell'.
                                               - 'quantity': Number of shares traded.
                                               - 'price': Price per share at which the trade occurred.
                                               - 'commission': Commission paid for the trade.
                                               - 'net_cash_impact': The actual change in cash balance due to the trade.
                                               - 'current_cash': Cash balance after the trade.
        commission_rate (float): The commission rate applied to trades (e.g., 0.001 for 0.1%).
    """

    def __init__(self, initial_cash: float, commission_rate: float = 0.001):
        """
        Initializes the PortfolioManager with a starting cash amount and commission rate.

        Args:
            initial_cash (float): The initial cash available in the portfolio.
            commission_rate (float): The commission rate as a decimal (e.g., 0.001 for 0.1%).
                                     Defaults to 0.1%.
        """
        if initial_cash < 0:
            raise ValueError("Initial cash cannot be negative.")
        if not (0 <= commission_rate < 1):
            raise ValueError("Commission rate must be between 0 and 1.")

        self.initial_cash: float = initial_cash
        self.cash: float = initial_cash
        self.positions: Dict[str, int] = {}  # {ticker: quantity}
        self.trade_history: List[Dict[str, Any]] = []
        self.commission_rate: float = commission_rate

        logger.info(f"PortfolioManager initialized with initial cash: ${initial_cash:,.2f}")
        logger.info(f"Commission rate set to: {self.commission_rate*100:.4f}%")

    def _record_trade(self, date: datetime, ticker: str, trade_type: str,
                      quantity: int, price: float, commission: float, net_cash_impact: float):
        """
        Records a trade in the history. This is an internal helper method.

        Args:
            date (datetime): The date and time the trade occurred.
            ticker (str): The ticker symbol of the asset.
            trade_type (str): 'buy' or 'sell'.
            quantity (int): The number of shares traded.
            price (float): The price per share.
            commission (float): The commission paid for this trade.
            net_cash_impact (float): The actual change in cash balance due to this trade.
        """
        trade_record = {
            "date": date,
            "ticker": ticker,
            "type": trade_type,
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "net_cash_impact": net_cash_impact,
            "current_cash": self.cash, # Snapshot of cash after the trade
            "current_positions": self.positions.copy() # Snapshot of positions after the trade
        }
        self.trade_history.append(trade_record)
        logger.debug(f"Trade recorded: {trade_record}")

    def buy(self, date: datetime, ticker: str, quantity: int, price: float) -> bool:
        """
        Executes a buy order for a specified quantity of shares at a given price.

        Args:
            date (datetime): The date and time of the trade.
            ticker (str): The ticker symbol of the asset to buy.
            quantity (int): The number of shares to buy. Must be positive.
            price (float): The price per share. Must be positive.

        Returns:
            bool: True if the buy order was successful, False otherwise (e.g., insufficient cash).
        """
        if quantity <= 0:
            logger.warning(f"Attempted to buy non-positive quantity ({quantity}) of {ticker} on {date.strftime('%Y-%m-%d')}.")
            return False
        if price <= 0:
            logger.warning(f"Attempted to buy {ticker} at non-positive price (${price:.2f}) on {date.strftime('%Y-%m-%d')}.")
            return False

        cost = quantity * price
        commission = cost * self.commission_rate
        total_cost = cost + commission

        if self.cash >= total_cost:
            self.cash -= total_cost
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
            self._record_trade(date, ticker, "buy", quantity, price, commission, -total_cost)
            logger.info(f"BUY {quantity} of {ticker} at ${price:.2f} on {date.strftime('%Y-%m-%d')}. "
                        f"Total cost: ${total_cost:,.2f} (incl. commission ${commission:,.2f}). "
                        f"Remaining cash: ${self.cash:,.2f}")
            return True
        else:
            logger.warning(f"Insufficient cash to buy {quantity} of {ticker} at ${price:.2f} on {date.strftime('%Y-%m-%d')}. "
                           f"Needed: ${total_cost:,.2f}, Available: ${self.cash:,.2f}")
            return False

    def sell(self, date: datetime, ticker: str, quantity: int, price: float) -> bool:
        """
        Executes a sell order for a specified quantity of shares at a given price.

        Args:
            date (datetime): The date and time of the trade.
            ticker (str): The ticker symbol of the asset to sell.
            quantity (int): The number of shares to sell. Must be positive.
            price (float): The price per share. Must be positive.

        Returns:
            bool: True if the sell order was successful, False otherwise (e.g., insufficient positions).
        """
        if quantity <= 0:
            logger.warning(f"Attempted to sell non-positive quantity ({quantity}) of {ticker} on {date.strftime('%Y-%m-%d')}.")
            return False
        if price <= 0:
            logger.warning(f"Attempted to sell {ticker} at non-positive price (${price:.2f}) on {date.strftime('%Y-%m-%d')}.")
            return False

        current_holding = self.positions.get(ticker, 0)
        if current_holding >= quantity:
            revenue = quantity * price
            commission = revenue * self.commission_rate
            net_revenue = revenue - commission

            self.cash += net_revenue
            self.positions[ticker] -= quantity
            if self.positions[ticker] == 0:
                del self.positions[ticker] # Remove ticker from positions if quantity becomes zero

            self._record_trade(date, ticker, "sell", quantity, price, commission, net_revenue)
            logger.info(f"SELL {quantity} of {ticker} at ${price:.2f} on {date.strftime('%Y-%m-%d')}. "
                        f"Net revenue: ${net_revenue:,.2f} (after commission ${commission:,.2f}). "
                        f"Remaining cash: ${self.cash:,.2f}")
            return True
        else:
            logger.warning(f"Insufficient {ticker} to sell {quantity} at ${price:.2f} on {date.strftime('%Y-%m-%d')}. "
                           f"Holding: {current_holding}")
            return False

    def get_current_cash(self) -> float:
        """
        Returns the current cash balance in the portfolio.

        Returns:
            float: The current cash balance.
        """
        return self.cash

    def get_current_positions(self) -> Dict[str, int]:
        """
        Returns a copy of the current positions held in the portfolio.

        Returns:
            Dict[str, int]: A dictionary mapping ticker symbols to quantities.
        """
        return self.positions.copy()

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Returns a copy of the complete trade history.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a recorded trade.
        """
        return self.trade_history.copy()

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculates the total value of the portfolio, including cash and the market
        value of all held positions.

        Args:
            current_prices (Dict[str, float]): A dictionary mapping ticker symbols
                                                to their current market prices.

        Returns:
            float: The total calculated portfolio value.
        """
        market_value = 0.0
        for ticker, quantity in self.positions.items():
            if ticker in current_prices:
                market_value += quantity * current_prices[ticker]
            else:
                # Log a warning if a price for a held asset is not provided
                logger.warning(f"Price for '{ticker}' not available when calculating portfolio value. "
                               f"Assuming 0 for this asset's contribution to market value.")
        return self.cash + market_value

    def get_initial_cash(self) -> float:
        """
        Returns the initial cash amount the portfolio was started with.

        Returns:
            float: The initial cash.
        """
        return self.initial_cash

    def reset(self):
        """
        Resets the portfolio to its initial state (initial cash, no positions, empty trade history).
        """
        self.cash = self.initial_cash
        self.positions = {}
        self.trade_history = []
        logger.info("PortfolioManager reset to initial state.")