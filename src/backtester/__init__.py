"""
Initializes the 'backtester' subpackage.

This subpackage forms the core of the backtesting engine, responsible for:
- Orchestrating the backtesting process.
- Simulating trades based on strategy signals.
- Managing portfolio state (cash, positions).
- Calculating performance metrics.
- Generating reports and visualizations.
"""

import logging

# Configure logging for the 'backtester' package
# This ensures that any module within the backtester package can get a logger
# that is a child of the 'backtester' logger, allowing for centralized configuration.
logger = logging.getLogger(__name__)
# Example: logger.addHandler(logging.StreamHandler()) # Add a handler if needed at package level

# Import core components to make them directly accessible from the 'backtester' package
# For example, if a main Backtester class is defined in backtester/engine.py:
# from .engine import Backtester
# from .portfolio import PortfolioManager
# from .metrics import PerformanceMetrics
# from .reporter import BacktestReporter

# As the specific files for these components are not yet defined,
# these imports are commented out as placeholders for future development.
# They would typically be uncommented or added as the corresponding files are created.

# Example of a package-level variable (optional)
# __all__ = ['Backtester', 'PortfolioManager', 'PerformanceMetrics', 'BacktestReporter']
# This defines what is imported when a user does `from src.backtester import *`