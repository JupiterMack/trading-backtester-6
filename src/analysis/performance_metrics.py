import pandas as pd
import numpy as np
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Calculates key performance indicators (KPIs) from backtest results.

    This class takes the portfolio's equity curve and a detailed trade log
    to compute various financial performance metrics such as total return,
    CAGR, volatility, Sharpe ratio, maximum drawdown, win rate, etc.
    """

    def __init__(self, equity_curve: pd.Series, trade_log: pd.DataFrame, annualization_factor: int = 252):
        """
        Initializes the PerformanceMetrics calculator with the equity curve and trade log.

        Args:
            equity_curve (pd.Series): A pandas Series where the index is a DatetimeIndex
                                      representing dates/times and values are the portfolio's
                                      total equity value at that point.
            trade_log (pd.DataFrame): A pandas DataFrame containing details of each trade.
                                      It must include a 'Profit/Loss' column indicating
                                      the P&L for each closed trade.
            annualization_factor (int): The number of periods in a year used for annualizing
                                        metrics (e.g., 252 for trading days, 365 for calendar days).
        """
        if not isinstance(equity_curve, pd.Series):
            raise TypeError("equity_curve must be a pandas Series.")
        if not isinstance(trade_log, pd.DataFrame):
            raise TypeError("trade_log must be a pandas DataFrame.")
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            logger.warning("Equity curve index is not a DatetimeIndex. Date-based calculations (e.g., CAGR) may be inaccurate.")

        if equity_curve.empty:
            logger.warning("Equity curve is empty. Performance metrics may not be meaningful.")
        if trade_log.empty:
            logger.warning("Trade log is empty. Trade-based metrics will not be calculated.")

        self.equity_curve = equity_curve.astype(float).sort_index()
        self.trade_log = trade_log.copy()
        self.annualization_factor = annualization_factor

        # Calculate daily/period returns from the equity curve
        # We use .iloc[0] for the initial value to avoid issues if the first index is not 0
        self.returns = self.equity_curve.pct_change().dropna()
        if self.returns.empty and len(self.equity_curve) > 1:
            logger.warning("Could not calculate returns from the equity curve. Check for constant values or invalid data.")
        elif self.returns.empty and len(self.equity_curve) <= 1:
            logger.warning("Equity curve has too few data points to calculate returns.")

    def calculate_all_metrics(self, risk_free_rate: float = 0.02) -> dict:
        """
        Calculates all available performance metrics and returns them in a dictionary.

        Args:
            risk_free_rate (float): The annual risk-free rate (e.g., 0.02 for 2%)
                                    used in Sharpe and Sortino ratio calculations.

        Returns:
            dict: A dictionary containing all calculated performance metrics.
        """
        metrics = {}
        try:
            metrics['total_return'] = self.calculate_total_return()
            metrics['cagr'] = self.calculate_cagr()
            metrics['annualized_volatility'] = self.calculate_annualized_volatility()
            metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(risk_free_rate)
            metrics['max_drawdown'] = self.calculate_max_drawdown()
            metrics['calmar_ratio'] = self.calculate_calmar_ratio()
            metrics['sortino_ratio'] = self.calculate_sortino_ratio(risk_free_rate)
            metrics['win_rate'] = self.calculate_win_rate()
            metrics['avg_winning_trade'] = self.calculate_average_winning_trade()
            metrics['avg_losing_trade'] = self.calculate_average_losing_trade()
            metrics['profit_factor'] = self.calculate_profit_factor()
            metrics['total_trades'] = len(self.trade_log)
        except Exception as e:
            logger.error(f"An error occurred during metric calculation: {e}", exc_info=True)
            # Depending on desired behavior, could re-raise, return partial, or return empty
            raise # Re-raise to indicate a critical failure in calculation

        return metrics

    def calculate_total_return(self) -> float:
        """
        Calculates the total percentage return over the backtesting period.
        Return = (Final Value - Initial Value) / Initial Value
        """
        if self.equity_curve.empty:
            return 0.0
        initial_value = self.equity_curve.iloc[0]
        final_value = self.equity_curve.iloc[-1]
        
        if initial_value == 0:
            logger.warning("Initial portfolio value is zero. Total return cannot be calculated.")
            return 0.0
        return (final_value - initial_value) / initial_value

    def calculate_cagr(self) -> float:
        """
        Calculates the Compound Annual Growth Rate (CAGR).
        CAGR = (End_Value / Start_Value)^(1 / Num_Years) - 1
        """
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        
        initial_value = self.equity_curve.iloc[0]
        final_value = self.equity_curve.iloc[-1]

        if initial_value == 0:
            logger.warning("Initial portfolio value is zero. CAGR cannot be calculated.")
            return 0.0

        # Calculate number of years in the backtest period
        # Ensure index is DatetimeIndex for accurate date difference
        if not isinstance(self.equity_curve.index, pd.DatetimeIndex):
            logger.warning("Equity curve index is not DatetimeIndex. Cannot accurately calculate CAGR.")
            return 0.0 # Or raise an error if strictness is required

        num_years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        
        if num_years <= 0:
            logger.warning("Backtest duration is too short or invalid for CAGR calculation.")
            return 0.0

        cagr = (final_value / initial_value)**(1 / num_years) - 1
        return cagr

    def calculate_annualized_volatility(self) -> float:
        """
        Calculates the annualized standard deviation of daily/period returns.
        """
        if self.returns.empty:
            return 0.0
        return self.returns.std() * np.sqrt(self.annualization_factor)

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculates the Sharpe Ratio.
        Sharpe Ratio = (Annualized Return - Risk-Free Rate) / Annualized Volatility
        """
        cagr = self.calculate_cagr()
        annual_volatility = self.calculate_annualized_volatility()

        if annual_volatility == 0:
            # If volatility is zero, Sharpe ratio is infinite if return is positive, else 0.
            return np.inf if cagr > risk_free_rate else 0.0
        
        return (cagr - risk_free_rate) / annual_volatility

    def calculate_max_drawdown(self) -> float:
        """
        Calculates the maximum drawdown percentage from the equity curve.
        Drawdown = (Trough Value - Peak Value) / Peak Value
        """
        if self.equity_curve.empty:
            return 0.0

        # Calculate cumulative maximum wealth (peak)
        cumulative_max = self.equity_curve.cummax()
        
        # Calculate drawdown (percentage drop from peak)
        drawdown = (self.equity_curve - cumulative_max) / cumulative_max
        
        # Max drawdown is the minimum (most negative) drawdown value
        max_drawdown = drawdown.min()
        return max_drawdown

    def calculate_calmar_ratio(self) -> float:
        """
        Calculates the Calmar Ratio.
        Calmar Ratio = CAGR / |Max Drawdown|
        """
        cagr = self.calculate_cagr()
        max_drawdown = self.calculate_max_drawdown()

        if max_drawdown == 0:
            # If no drawdown, Calmar ratio is infinite if CAGR is positive, else 0.
            return np.inf if cagr > 0 else 0.0
        
        return cagr / abs(max_drawdown)

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculates the Sortino Ratio.
        Sortino Ratio = (Annualized Return - Risk-Free Rate) / Downside Deviation
        Downside deviation only considers negative returns.
        """
        if self.returns.empty:
            return 0.0

        # Convert annual risk-free rate to daily/period equivalent for comparison with daily returns
        daily_risk_free_rate = (1 + risk_free_rate)**(1/self.annualization_factor) - 1
        
        # Calculate excess returns relative to the daily risk-free rate
        excess_returns = self.returns - daily_risk_free_rate

        # Identify downside returns (returns below the daily risk-free rate)
        downside_returns = excess_returns[excess_returns < 0]

        if downside_returns.empty:
            # No downside volatility, Sortino is infinite if return is positive, else 0.
            return np.inf if self.calculate_cagr() > risk_free_rate else 0.0

        # Calculate downside deviation (annualized)
        downside_deviation = downside_returns.std() * np.sqrt(self.annualization_factor)

        if downside_deviation == 0:
            # If downside deviation is zero, Sortino is infinite if return is positive, else 0.
            return np.inf if self.calculate_cagr() > risk_free_rate else 0.0

        cagr = self.calculate_cagr()
        return (cagr - risk_free_rate) / downside_deviation

    def calculate_win_rate(self) -> float:
        """
        Calculates the win rate based on the trade log.
        Win Rate = (Number of Profitable Trades) / (Total Number of Trades)
        Assumes 'Profit/Loss' column exists in trade_log.
        """
        if self.trade_log.empty or 'Profit/Loss' not in self.trade_log.columns:
            logger.warning("Trade log is empty or missing 'Profit/Loss' column for win rate calculation.")
            return 0.0
        
        profitable_trades = self.trade_log[self.trade_log['Profit/Loss'] > 0]
        total_trades = len(self.trade_log)
        
        if total_trades == 0:
            return 0.0
        return len(profitable_trades) / total_trades

    def calculate_average_winning_trade(self) -> float:
        """
        Calculates the average profit of winning trades.
        """
        if self.trade_log.empty or 'Profit/Loss' not in self.trade_log.columns:
            return 0.0
        
        winning_trades = self.trade_log[self.trade_log['Profit/Loss'] > 0]
        if winning_trades.empty:
            return 0.0
        return winning_trades['Profit/Loss'].mean()

    def calculate_average_losing_trade(self) -> float:
        """
        Calculates the average loss of losing trades.
        """
        if self.trade_log.empty or 'Profit/Loss' not in self.trade_log.columns:
            return 0.0
        
        losing_trades = self.trade_log[self.trade_log['Profit/Loss'] < 0]
        if losing_trades.empty:
            return 0.0
        return losing_trades['Profit/Loss'].mean()

    def calculate_profit_factor(self) -> float:
        """
        Calculates the Profit Factor (Gross Profit / Gross Loss).
        Gross Profit = Sum of profits from winning trades.
        Gross Loss = Sum of losses from losing trades (absolute value).
        """
        if self.trade_log.empty or 'Profit/Loss' not in self.trade_log.columns:
            return 0.0
        
        gross_profit = self.trade_log[self.trade_log['Profit/Loss'] > 0]['Profit/Loss'].sum()
        gross_loss = self.trade_log[self.trade_log['Profit/Loss'] < 0]['Profit/Loss'].sum()
        
        if gross_loss == 0:
            # If no losses, profit factor is infinite if there's profit, else 0.
            return np.inf if gross_profit > 0 else 0.0
        
        return abs(gross_profit / gross_loss)