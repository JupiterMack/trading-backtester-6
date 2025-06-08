"""
Makes the 'src' directory a Python package.

This file can be used to define package-level variables,
such as the package version, or to perform package-wide
initialization.
"""

# Define the package version, consistent with pyproject.toml
__version__ = "0.1.0"

# You can optionally import submodules here to make them directly
# accessible under the 'src' namespace, e.g., 'from src import data'.
# However, for clarity and to avoid circular dependencies, it's often
# better to let users import specific modules directly, e.g.,
# 'from src.data import yfinance_handler'.

# No specific imports or code needed here beyond package metadata
# for a basic package definition.
# More complex packages might use this file for:
# - Importing core components for easier access (e.g., from . import utils)
# - Setting up logging configuration
# - Defining package-wide constants or exceptions
# - Registering plugins or extensions
# For this project, a simple package definition is sufficient.