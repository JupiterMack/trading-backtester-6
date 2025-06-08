import abc
from abc import ABC, abstractmethod
import pandas as pd
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract Base Class (ABC) for all trading strategies.

    This class defines the interface that all concrete trading strategies
    must implement. It ensures a consistent structure for strategy
    development within the backtesting framework.

    Each strategy will typically take historical data (which may already
    include technical indicators processed by DataProcessor) and generate
    trading signals.
    """

    def __init__(self, name: str, parameters: dict = None):
        """
        Initializes the base strategy with a name and optional parameters.

        Args:
            name (str): The unique name of the strategy (e.g., "MovingAverageCrossover").
            parameters (dict, optional): A dictionary of parameters specific
                                         to this strategy instance. These parameters
                                         will be used by the strategy's logic
                                         (e.g., period for moving averages).
                                         Defaults to an empty dictionary if None.
        Raises:
            ValueError: If the strategy name is not a non-empty string.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Strategy name must be a non-empty string.")
        self._name = name.strip()
        self._parameters = parameters if parameters is not None else {}
        logger.info(f"Initialized strategy: '{self._name}' with parameters: {self._parameters}")

    @property
    def name(self) -> str:
        """
        Returns the name of the strategy.

        Returns:
            str: The strategy's name.
        """
        return self._name

    @property
    def parameters(self) -> dict:
        """
        Returns the parameters dictionary of the strategy.

        Returns:
            dict: A dictionary containing the strategy's specific parameters.
        """
        return self._parameters

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to generate trading signals based on historical data.

        Concrete strategy implementations must override this method to define
        their specific signal generation logic. The method should add new
        columns to the input DataFrame representing the trading signals.

        Common signal columns might include:
        - 'signal': Numeric (e.g., 1 for buy, -1 for sell, 0 for hold)
        - 'entry_signal': Boolean (True for a long entry)
        - 'exit_signal': Boolean (True for a long exit)
        - 'short_entry_signal': Boolean (True for a short entry)
        - 'short_exit_signal': Boolean (True for a short exit)

        It is crucial that the implementation handles any required technical
        indicator calculations if they are not already present in the 'data'
        DataFrame, or assumes that 'data' has been pre-processed by a
        DataProcessor to include necessary indicators.

        Args:
            data (pd.DataFrame): A DataFrame containing historical OHLCV data
                                 and any pre-processed indicators.
                                 Expected columns typically include 'Open', 'High',
                                 'Low', 'Close', 'Volume', and potentially
                                 technical indicators. The DataFrame index
                                 is expected to be a DatetimeIndex.

        Returns:
            pd.DataFrame: The input DataFrame with additional columns for
                          trading signals. The original data columns should
                          be preserved.
        """
        pass

    def __str__(self) -> str:
        """
        Returns a string representation of the strategy.
        """
        return f"{self._name} Strategy (Params: {self._parameters})"

    def __repr__(self) -> str:
        """
        Returns a developer-friendly string representation of the strategy.
        """
        return f"<{self.__class__.__name__}(name='{self._name}', parameters={self._parameters})>"