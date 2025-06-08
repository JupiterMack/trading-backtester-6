import logging
import os
import configparser
from datetime import datetime
import math

# Define a logger for this module itself.
# This logger is used for messages specific to the helpers module,
# while the root logger (configured by setup_logging) handles general application logs.
logger = logging.getLogger(__name__)


def setup_logging(log_level_str: str = 'INFO', log_file: str = None, log_dir: str = 'logs'):
    """
    Configures the logging for the entire application.

    Sets up a console handler and an optional file handler.
    Removes existing handlers from the root logger to prevent duplicate log messages
    if this function is called multiple times.

    Args:
        log_level_str (str): The desired logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_file (str, optional): Name of the log file (e.g., 'backtest.log'). If None, file logging is disabled.
        log_dir (str): Directory where log files will be stored if `log_file` is provided.
                       This directory will be created if it does not exist.
    """
    # Map string level to logging constants
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove all existing handlers to prevent duplicate logs if called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Define a common formatter for all handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add a console handler to output logs to the standard output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add a file handler if a log file name is specified
    if log_file:
        try:
            # Ensure the log directory exists
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            file_path = os.path.join(log_dir, log_file)
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"Logging to file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to set up file logging: {e}")

    logger.info(f"Root logging level set to: {log_level_str.upper()}")


def get_config_value(config: configparser.ConfigParser, section: str, option: str, default=None, val_type=str):
    """
    Safely retrieves a configuration value from a ConfigParser object,
    handling missing sections/options and performing type conversion.

    Args:
        config (configparser.ConfigParser): The ConfigParser object containing the configuration.
        section (str): The name of the section in the configuration file (e.g., 'DATA', 'STRATEGY').
        option (str): The name of the option within the specified section (e.g., 'data_source', 'initial_capital').
        default: The default value to return if the option or section is not found.
                 If None and the option/section is missing, a `configparser` error is raised.
        val_type (type): The desired type for the retrieved value (e.g., str, int, float, bool).

    Returns:
        The retrieved and type-converted value, or the default value if specified and the option is missing.

    Raises:
        configparser.NoSectionError: If the section does not exist and no default is provided.
        configparser.NoOptionError: If the option does not exist within the section and no default is provided.
        ValueError: If the value cannot be converted to the specified `val_type` and no default is provided.
    """
    try:
        if val_type == int:
            value = config.getint(section, option)
        elif val_type == float:
            value = config.getfloat(section, option)
        elif val_type == bool:
            # configparser.getboolean is robust and handles 'yes', 'no', 'on', 'off', 'true', 'false', '1', '0'
            value = config.getboolean(section, option)
        else:  # Default to string
            value = config.get(section, option)
        return value
    except (configparser.NoSectionError, configparser.NoOptionError):
        if default is not None:
            logger.warning(f"Config option '{option}' not found in section '{section}'. Using default: {default}")
            return default
        else:
            logger.error(f"Config option '{option}' not found in section '{section}' and no default provided.")
            raise  # Re-raise original configparser error if no default
    except ValueError as e:
        logger.error(f"Error converting config option '{option}' in section '{section}' to type {val_type.__name__}: {e}")
        if default is not None:
            logger.warning(f"Using default value {default} due to type conversion error.")
            return default
        else:
            raise  # Re-raise original ValueError if no default


def parse_date(date_str: str, date_format: str = '%Y-%m-%d') -> datetime:
    """
    Parses a date string into a datetime object.

    Args:
        date_str (str): The date string to parse (e.g., '2023-01-15', '01/15/2023').
        date_format (str): The format of the date string (e.g., '%Y-%m-%d', '%m/%d/%Y').
                           Refer to `datetime.strptime` documentation for format codes.

    Returns:
        datetime: The parsed datetime object.

    Raises:
        ValueError: If the date string does not match the specified format or is invalid.
    """
    try:
        return datetime.strptime(date_str, date_format)
    except ValueError as e:
        logger.error(f"Failed to parse date string '{date_str}' with format '{date_format}': {e}")
        raise


def calculate_annualized_return(total_return: float, num_years: float) -> float:
    """
    Calculates the annualized return from a total return over a period of years.

    Formula: (1 + Total Return)^(1 / Number of Years) - 1

    Args:
        total_return (float): The total return over the period (e.g., 0.50 for 50% return, -0.20 for -20% return).
        num_years (float): The number of years over which the total return was achieved.
                           Can be fractional (e.g., 0.5 for 6 months).

    Returns:
        float: The annualized return. Returns 0.0 if `num_years` is 0 or less.
               Returns -1.0 if the total return indicates a loss greater than 100%
               (i.e., `1 + total_return` is negative), as this typically means
               the portfolio is wiped out.
    """
    if num_years <= 0:
        logger.warning("Number of years must be positive for annualized return calculation. Returning 0.0.")
        return 0.0

    base = 1 + total_return
    if base < 0:
        logger.warning(f"Base for annualized return calculation (1 + total_return = {base}) is negative. "
                       "This typically indicates a loss greater than 100%. Returning -1.0 (100% loss).")
        return -1.0  # Portfolio is wiped out

    return math.pow(base, 1 / num_years) - 1


if __name__ == '__main__':
    # This block demonstrates the usage of the helper functions when the file is run directly.
    # It's useful for testing and verifying the functionality of each helper.

    # --- Test setup_logging ---
    print("--- Testing setup_logging ---")
    # Ensure 'logs' directory exists for testing file logging
    if not os.path.exists('logs'):
        os.makedirs('logs')
    setup_logging(log_level_str='DEBUG', log_file='test_helpers.log')
    logger.debug("This is a debug message from helpers.py (should appear in console and file)")
    logger.info("This is an info message from helpers.py (should appear in console and file)")
    logger.warning("This is a warning message from helpers.py (should appear in console and file)")
    logger.error("This is an error message from helpers.py (should appear in console and file)")

    # --- Test get_config_value ---
    print("\n--- Testing get_config_value ---")
    test_config = configparser.ConfigParser()
    test_config['TEST_SECTION'] = {
        'string_val': 'hello world',
        'int_val': '123',
        'float_val': '45.67',
        'bool_true': 'True',
        'bool_false': '0',
        'empty_val': ''
    }
    test_config['ANOTHER_SECTION'] = {
        'some_option': 'value'
    }

    # Existing values with type conversion
    s_val = get_config_value(test_config, 'TEST_SECTION', 'string_val')
    i_val = get_config_value(test_config, 'TEST_SECTION', 'int_val', val_type=int)
    f_val = get_config_value(test_config, 'TEST_SECTION', 'float_val', val_type=float)
    b_true = get_config_value(test_config, 'TEST_SECTION', 'bool_true', val_type=bool)
    b_false = get_config_value(test_config, 'TEST_SECTION', 'bool_false', val_type=bool)
    empty_val = get_config_value(test_config, 'TEST_SECTION', 'empty_val')
    print(f"String value: '{s_val}' (type: {type(s_val)})")
    print(f"Int value: {i_val} (type: {type(i_val)})")
    print(f"Float value: {f_val} (type: {type(f_val)})")
    print(f"Bool true value: {b_true} (type: {type(b_true)})")
    print(f"Bool false value: {b_false} (type: {type(b_false)})")
    print(f"Empty value: '{empty_val}' (type: {type(empty_val)})")

    # Missing values with default
    missing_str = get_config_value(test_config, 'TEST_SECTION', 'non_existent_str', default='default_string')
    missing_int = get_config_value(test_config, 'TEST_SECTION', 'non_existent_int', default=999, val_type=int)
    print(f"Missing string (default): {missing_str}")
    print(f"Missing int (default): {missing_int}")

    # Missing section/option without default (should raise error)
    try:
        print("\nAttempting to get value from non-existent section without default...")
        get_config_value(test_config, 'NON_EXISTENT_SECTION', 'some_option')
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"Caught expected error for missing section/option without default: {e}")

    # Type conversion error without default (should raise error)
    test_config['TEST_SECTION']['bad_int'] = 'not_an_integer'
    try:
        print("\nAttempting to convert 'not_an_integer' to int without default...")
        get_config_value(test_config, 'TEST_SECTION', 'bad_int', val_type=int)
    except ValueError as e:
        print(f"Caught expected error for type conversion: {e}")
    
    # Type conversion error with default
    bad_int_default = get_config_value(test_config, 'TEST_SECTION', 'bad_int', default=0, val_type=int)
    print(f"Bad int with default (should use default): {bad_int_default}")


    # --- Test parse_date ---
    print("\n--- Testing parse_date ---")
    date_str1 = "2023-01-15"
    date_obj1 = parse_date(date_str1)
    print(f"Parsed '{date_str1}': {date_obj1} (type: {type(date_obj1)})")

    date_str2 = "01/15/2023"
    date_obj2 = parse_date(date_str2, date_format='%m/%d/%Y')
    print(f"Parsed '{date_str2}' with custom format: {date_obj2}")

    # Invalid date string (should raise ValueError)
    try:
        print("\nAttempting to parse invalid date '2023-13-01'...")
        parse_date("2023-13-01")  # Invalid month
    except ValueError as e:
        print(f"Caught expected error for invalid date: {e}")

    try:
        print("\nAttempting to parse malformed date string 'not-a-date'...")
        parse_date("not-a-date")
    except ValueError as e:
        print(f"Caught expected error for malformed date string: {e}")

    # --- Test calculate_annualized_return ---
    print("\n--- Testing calculate_annualized_return ---")
    # 100% return over 1 year
    ann_ret1 = calculate_annualized_return(1.0, 1)
    print(f"Total 100% over 1 year: {ann_ret1:.4f} (Expected: 1.0000)")

    # 100% return over 2 years (approx 41.42% annually)
    ann_ret2 = calculate_annualized_return(1.0, 2)
    print(f"Total 100% over 2 years: {ann_ret2:.4f} (Expected: 0.4142)")

    # 50% return over 0.5 years (approx 125% annually)
    ann_ret3 = calculate_annualized_return(0.5, 0.5)
    print(f"Total 50% over 0.5 years: {ann_ret3:.4f} (Expected: 1.2500)")

    # -50% return over 1 year
    ann_ret4 = calculate_annualized_return(-0.5, 1)
    print(f"Total -50% over 1 year: {ann_ret4:.4f} (Expected: -0.5000)")

    # -50% return over 2 years (approx -29.29% annually)
    ann_ret5 = calculate_annualized_return(-0.5, 2)
    print(f"Total -50% over 2 years: {ann_ret5:.4f} (Expected: -0.2929)")

    # Edge case: num_years = 0 or negative
    ann_ret_zero_years = calculate_annualized_return(0.5, 0)
    print(f"Total 50% over 0 years: {ann_ret_zero_years:.4f} (Expected: 0.0000)")
    ann_ret_neg_years = calculate_annualized_return(0.5, -1)
    print(f"Total 50% over -1 years: {ann_ret_neg_years:.4f} (Expected: 0.0000)")

    # Edge case: Total loss > 100% (e.g., -150% return)
    ann_ret_huge_loss = calculate_annualized_return(-1.5, 1)
    print(f"Total -150% over 1 year: {ann_ret_huge_loss:.4f} (Expected: -1.0000)")
    ann_ret_huge_loss_multi_year = calculate_annualized_return(-1.5, 2)
    print(f"Total -150% over 2 years: {ann_ret_huge_loss_multi_year:.4f} (Expected: -1.0000)")


    # Clean up test log file and directory after tests
    try:
        if os.path.exists('logs/test_helpers.log'):
            os.remove('logs/test_helpers.log')
        if os.path.exists('logs'):
            os.rmdir('logs')
        print("\nCleaned up test log file and directory.")
    except OSError as e:
        print(f"Error during cleanup: {e}")