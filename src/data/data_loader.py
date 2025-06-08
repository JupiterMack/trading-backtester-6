import pandas as pd
import yfinance as yf
import os
import logging
import configparser
from datetime import datetime

# Configure logging for this module
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading historical stock data from various sources (e.g., CSV, API)
    into pandas DataFrames.

    The loaded DataFrame will have a DatetimeIndex named 'Date' and
    standardized columns: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.
    """

    def __init__(self, config: configparser.ConfigParser):
        """
        Initializes the DataLoader with project configuration.

        Args:
            config (configparser.ConfigParser): The loaded configuration object.
        """
        self.config = config
        self.data_source = self.config.get('DATA', 'data_source', fallback='yfinance').lower()
        self.csv_file_path = self.config.get('DATA', 'csv_file_path', fallback=None)
        logger.info(f"DataLoader initialized with data source: '{self.data_source}'")
        if self.data_source == 'local_csv':
            logger.info(f"CSV file path configured: '{self.csv_file_path}'")

    def load_data(self, ticker: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Loads historical stock data based on the configured data source.

        Args:
            ticker (str, optional): The stock ticker symbol (e.g., 'AAPL'). Required for 'yfinance' source.
            start_date (str, optional): Start date for data in 'YYYY-MM-DD' format. Required for 'yfinance' source.
            end_date (str, optional): End date for data in 'YYYY-MM-DD' format. Required for 'yfinance' source.

        Returns:
            pd.DataFrame: A DataFrame containing historical stock data with a DatetimeIndex.
                          Expected columns: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.

        Raises:
            ValueError: If required parameters are missing, data source is unsupported,
                        or data format is incorrect.
            FileNotFoundError: If a local CSV file is not found.
            Exception: For other data loading errors.
        """
        if self.data_source == 'yfinance':
            if not all([ticker, start_date, end_date]):
                raise ValueError("For 'yfinance' data source, 'ticker', 'start_date', and 'end_date' must be provided.")
            return self._load_from_yfinance(ticker, start_date, end_date)
        elif self.data_source == 'local_csv':
            if not self.csv_file_path:
                raise ValueError("For 'local_csv' data source, 'csv_file_path' must be specified in config.ini.")
            return self._load_from_csv(self.csv_file_path)
        else:
            raise ValueError(f"Unsupported data source: '{self.data_source}'. "
                             "Please choose 'yfinance' or 'local_csv' in config.ini.")

    def _load_from_yfinance(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Loads historical data for a given ticker from Yahoo Finance.

        Args:
            ticker (str): The stock ticker symbol.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: Historical data with DatetimeIndex.

        Raises:
            ValueError: If no data is found or required columns are missing.
            Exception: If data download fails.
        """
        logger.info(f"Attempting to load data for {ticker} from Yahoo Finance "
                    f"from {start_date} to {end_date}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df.empty:
                logger.warning(f"No data found for {ticker} between {start_date} and {end_date}.")
                raise ValueError(f"No data found for {ticker} in the specified date range.")

            # Ensure 'Adj Close' is present. If not, use 'Close'.
            if 'Adj Close' not in df.columns:
                if 'Close' in df.columns:
                    df['Adj Close'] = df['Close']
                    logger.warning(f"No 'Adj Close' column found for {ticker}. Using 'Close' as 'Adj Close'.")
                else:
                    raise ValueError(f"Neither 'Adj Close' nor 'Close' column found in data for {ticker}.")

            # Standardize index and select required columns
            df.index.name = 'Date'
            required_yf_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            
            # Check if all required columns exist
            missing_cols = [col for col in required_yf_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Yahoo Finance data for {ticker} is missing required columns: {', '.join(missing_cols)}.")

            df = df[required_yf_cols]
            logger.info(f"Successfully loaded {len(df)} rows for {ticker} from Yahoo Finance.")
            return df
        except Exception as e:
            logger.error(f"Error loading data from Yahoo Finance for {ticker}: {e}")
            raise

    def _load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Loads historical data from a local CSV file.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: Historical data with DatetimeIndex.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
            ValueError: If the CSV file is malformed or missing required columns.
            Exception: For other CSV reading errors.
        """
        if not os.path.exists(file_path):
            logger.error(f"CSV file not found: {file_path}")
            raise FileNotFoundError(f"The specified CSV file does not exist: {file_path}")

        logger.info(f"Attempting to load data from local CSV: {file_path}...")
        try:
            df = pd.read_csv(file_path)

            # Standardize column names to a consistent format (e.g., 'Date', 'Open', 'High'...)
            # Create a mapping for common variations (case-insensitive, with/without spaces/underscores)
            col_map = {
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'adj close': 'Adj Close',
                'adj_close': 'Adj Close',
                'volume': 'Volume'
            }
            # Convert current columns to lowercase for matching and rename
            df.columns = [col.lower() for col in df.columns]
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            # Check for 'Date' column after renaming
            if 'Date' not in df.columns:
                raise ValueError("CSV file must contain a 'Date' column (case-insensitive, e.g., 'Date', 'date').")

            # Convert 'Date' column to datetime and set as index
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            df.index.name = 'Date'

            # Define the final set of required columns for the output DataFrame
            # 'Adj Close' is handled separately as it can be derived from 'Close'
            core_required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Check for core missing columns
            missing_cols = [col for col in core_required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV file is missing required columns: {', '.join(missing_cols)}. "
                                 f"Expected at least: {', '.join(core_required_cols)}.")

            # Handle 'Adj Close' specifically: if missing, try to use 'Close'
            if 'Adj Close' not in df.columns:
                if 'Close' in df.columns:
                    df['Adj Close'] = df['Close']
                    logger.warning(f"CSV missing 'Adj Close', using 'Close' for 'Adj Close'.")
                else:
                    # This case should ideally not happen if 'Close' is in core_required_cols
                    # and checked above, but as a safeguard.
                    raise ValueError("CSV file is missing both 'Adj Close' and 'Close' columns.")

            # Ensure all target columns are present before selecting and reordering
            final_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            
            # Select and reorder columns
            df = df[final_cols]

            # Ensure numerical types and drop rows with NaN in critical columns
            for col in final_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=final_cols, inplace=True) 

            if df.empty:
                logger.warning(f"CSV file '{file_path}' loaded but resulted in an empty DataFrame after processing.")
                raise ValueError(f"CSV file '{file_path}' resulted in no valid data after processing.")

            logger.info(f"Successfully loaded {len(df)} rows from local CSV: {file_path}.")
            return df
        except Exception as e:
            logger.error(f"Error loading data from CSV file '{file_path}': {e}")
            raise

# Example usage (for testing/demonstration)
if __name__ == '__main__':
    # Setup basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create a dummy config for testing
    test_config = configparser.ConfigParser()
    
    # --- Test Yahoo Finance loading ---
    print("\n--- Testing Yahoo Finance ---")
    test_config['DATA'] = {
        'data_source': 'yfinance',
        'csv_file_path': 'data/test_stock_data.csv' # This won't be used for yfinance test
    }
    data_loader_yf = DataLoader(test_config)
    try:
        # Using a fixed date range for testing
        yf_data = data_loader_yf.load_data(ticker='AAPL', start_date='2023-01-01', end_date='2023-01-31')
        print(f"AAPL data from Yahoo Finance (first 5 rows):\n{yf_data.head()}")
        print(f"AAPL data shape: {yf_data.shape}")
        print(f"AAPL data columns: {yf_data.columns.tolist()}")
        print(f"AAPL data index type: {type(yf_data.index)}")
    except Exception as e:
        print(f"Failed to load AAPL data from Yahoo Finance: {e}")

    # --- Test CSV loading ---
    print("\n--- Testing Local CSV ---")
    # Create a dummy CSV file for testing
    dummy_csv_dir = 'data'
    dummy_csv_path = os.path.join(dummy_csv_dir, 'test_stock_data.csv')
    os.makedirs(dummy_csv_dir, exist_ok=True)
    dummy_data = {
        'Date': pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
        'Open': [170.0, 172.0, 175.0, 173.0, 174.0],
        'High': [172.5, 175.0, 176.0, 174.5, 175.5],
        'Low': [169.0, 171.0, 173.0, 172.0, 173.0],
        'Close': [171.5, 174.0, 174.5, 173.5, 175.0],
        'Adj Close': [170.0, 172.5, 173.0, 172.0, 173.5],
        'Volume': [1000000, 1100000, 900000, 1200000, 950000]
    }
    pd.DataFrame(dummy_data).to_csv(dummy_csv_path, index=False)
    print(f"Created dummy CSV at: {dummy_csv_path}")

    test_config['DATA']['data_source'] = 'local_csv'
    test_config['DATA']['csv_file_path'] = dummy_csv_path
    data_loader_csv = DataLoader(test_config)

    try:
        csv_data = data_loader_csv.load_data() # No ticker/dates needed for CSV
        print(f"CSV data (first 5 rows):\n{csv_data.head()}")
        print(f"CSV data shape: {csv_data.shape}")
        print(f"CSV data columns: {csv_data.columns.tolist()}")
        print(f"CSV data index type: {type(csv_data.index)}")
    except Exception as e:
        print(f"Failed to load data from CSV: {e}")
    finally:
        # Clean up dummy CSV
        if os.path.exists(dummy_csv_path):
            os.remove(dummy_csv_path)
            print(f"Cleaned up dummy CSV: {dummy_csv_path}")
        if os.path.exists(dummy_csv_dir) and not os.listdir(dummy_csv_dir):
            os.rmdir(dummy_csv_dir) # Remove 'data' dir if empty

    # --- Test error handling (missing CSV) ---
    print("\n--- Testing Missing CSV Error ---")
    test_config['DATA']['csv_file_path'] = 'data/non_existent.csv'
    data_loader_missing_csv = DataLoader(test_config)
    try:
        data_loader_missing_csv.load_data()
    except FileNotFoundError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    # --- Test error handling (unsupported source) ---
    print("\n--- Testing Unsupported Source Error ---")
    test_config['DATA']['data_source'] = 'unsupported_source'
    data_loader_unsupported = DataLoader(test_config)
    try:
        data_loader_unsupported.load_data()
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    # --- Test error handling (yfinance missing params) ---
    print("\n--- Testing YFinance Missing Params Error ---")
    test_config['DATA']['data_source'] = 'yfinance'
    data_loader_yf_missing_params = DataLoader(test_config)
    try:
        data_loader_yf_missing_params.load_data(ticker='MSFT', start_date='2023-01-01') # Missing end_date
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")