"""
Initializes the 'strategies' subpackage.

This subpackage is designed to house various trading strategy implementations.
Each strategy typically defines:
- A method to generate trading signals (e.g., buy, sell, hold).
- Logic for managing positions based on these signals.

Common functionalities or helper classes shared across multiple strategies
might also be defined or imported here for easier access within the subpackage.
"""

# Although not strictly necessary for a basic __init__.py,
# one might choose to import common strategy components or base classes here
# if they are meant to be directly accessible from the 'strategies' package.
# For example:
# from .base_strategy import BaseStrategy
# from .utils import calculate_indicators

# For now, we'll keep it simple as the individual strategy files
# will handle their specific imports.