import pandas as pd
import numpy as np
import logging
import pandas_ta as ta

# Configure logging for this module
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles cleaning, transforming, and preparing raw historical data for backtesting.
    This includes standardizing column names, handling missing values,
    calculating returns, and adding technical indicators.
    """

    def __init__(self):
        """
        Initializes the DataProcessor.
        Currently, no specific parameters are needed for initialization,
        but this can be extended in the future.
        """
        pass

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes column names to a consistent format (e.g., 'Open', 'High', 'Low', 'Close', 'Volume').
        Prioritizes 'Adj Close' over 'Close' if both are present for the main price column.
        Keeps other non-standardized columns that might be present in the raw data.

        Args:
            df (pd.DataFrame): The input DataFrame with raw historical data.

        Returns:
            pd.DataFrame: DataFrame with standardized column names.
        
        Raises:
            ValueError: If neither 'Close' nor 'Adj Close' column is found.
        """
        df_copy = df.copy()
        
        # Convert column names to lowercase and replace spaces for easier matching
        df_copy.columns = [col.lower().replace(' ', '_') for col in df_copy.columns]

        # Define standard names and their possible raw mappings (lowercase)
        standard_name_map = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adj_close': 'Adj Close',
            'volume': 'Volume'
        }
        
        # Perform renaming based on the map
        renamed_cols = {}
        for raw_col_lower, standard_col_name in standard_name_map.items():
            if raw_col_lower in df_copy.columns:
                renamed_cols[raw_col_lower] = standard_col_name
        
        df_copy = df_copy.rename(columns=renamed_cols)

        # Handle 'Adj Close' vs 'Close' priority for the primary price column
        if 'Adj Close' in df_copy.columns and 'Close' in df_copy.columns:
            # If both exist, 'Adj Close' is generally preferred for backtesting
            # as it accounts for splits and dividends. We'll use it as 'Close'.
            df_copy['Close'] = df_copy['Adj Close']
            df_copy = df_copy.drop(columns=['Adj Close'])
            logger.info("Used 'Adj Close' as the primary 'Close' price and dropped original 'Close'.")
        elif 'Adj Close' in df_copy.columns:
            df_copy = df_copy.rename(columns={'Adj Close': 'Close'})
            logger.info("Renamed 'Adj Close' to 'Close'.")
        elif 'Close' not in df_copy.columns:
            logger.error("Neither 'Close' nor 'Adj Close' column found after standardization.")
            raise ValueError("Missing 'Close' or 'Adj Close' column in raw data. Cannot proceed.")

        # Ensure essential columns are present after renaming (warn if missing)
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df_copy.columns:
                logger.warning(f"Missing expected column: '{col}' after standardization. This might affect subsequent calculations.")
        
        # Reorder columns: standard ones first, then any other remaining original columns
        ordered_cols = []
        for col in required_cols:
            if col in df_copy.columns:
                ordered_cols.append(col)
        
        # Add any other columns that were not part of the standard set
        for col in df_copy.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)
        
        return df_copy[ordered_cols]

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by ensuring a proper DatetimeIndex, sorting,
        handling missing values, and removing duplicate entries.

        - Converts index to DatetimeIndex if not already.
        - Sorts the DataFrame by index (date).
        - Fills missing price data ('Open', 'High', 'Low', 'Close') using forward fill.
        - Fills missing 'Volume' with 0.
        - Drops any remaining rows with NaN values (e.g., NaNs at the start of the series).
        - Removes duplicate index entries, keeping the first occurrence.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        
        Raises:
            Exception: If the index cannot be converted to DatetimeIndex.
        """
        df_copy = df.copy()

        # Ensure index is DatetimeIndex and sorted
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            try:
                df_copy.index = pd.to_datetime(df_copy.index)
                logger.info("Converted DataFrame index to DatetimeIndex.")
            except Exception as e:
                logger.error(f"Failed to convert index to DatetimeIndex: {e}")
                raise
        
        df_copy = df_copy.sort_index()
        logger.info("DataFrame sorted by index (date).")

        initial_rows = len(df_copy)
        
        # Handle missing values:
        # For OHLC prices, forward fill is generally appropriate for gaps in time series.
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df_copy.columns and df_copy[col].isnull().any():
                logger.warning(f"Missing values found in '{col}'. Attempting forward fill.")
                df_copy[col] = df_copy[col].ffill()
        
        # For Volume, missing values can sometimes be treated as 0 (no trade) or dropped.
        # Filling with 0 is safer than dropping the whole row if only volume is missing.
        if 'Volume' in df_copy.columns and df_copy['Volume'].isnull().any():
            logger.warning("Missing values found in 'Volume'. Attempting to fill with 0.")
            df_copy['Volume'] = df_copy['Volume'].fillna(0)

        # Drop any remaining rows with NaN values (e.g., if ffill couldn't fill initial NaNs)
        if df_copy.isnull().any().any():
            nan_rows_before_drop = df_copy.isnull().any(axis=1).sum()
            df_copy.dropna(inplace=True)
            if nan_rows_before_drop > 0:
                logger.warning(f"Dropped {nan_rows_before_drop} rows due to remaining NaN values after filling.")

        # Check for duplicate indices (dates) and remove them
        if not df_copy.index.is_unique:
            logger.warning("Duplicate dates found in data. Dropping duplicates, keeping the first entry.")
            df_copy = df_copy[~df_copy.index.duplicated(keep='first')]
            logger.info(f"Removed duplicate dates. Remaining rows: {len(df_copy)}")

        final_rows = len(df_copy)
        if initial_rows != final_rows:
            logger.info(f"Cleaned data: {initial_rows - final_rows} rows removed. Remaining rows: {final_rows}")
        else:
            logger.info("No rows removed during cleaning process.")
        
        return df_copy

    def _calculate_returns(self, df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
        """
        Calculates daily percentage returns for the specified price column.
        Adds a new column named 'Daily_Return' to the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            price_col (str): The name of the column to use for return calculation (default: 'Close').

        Returns:
            pd.DataFrame: DataFrame with the 'Daily_Return' column added.
        
        Raises:
            ValueError: If the specified price column is not found.
        """
        df_copy = df.copy()
        if price_col not in df_copy.columns:
            logger.error(f"Price column '{price_col}' not found for return calculation.")
            raise ValueError(f"Price column '{price_col}' not found in DataFrame.")
        
        df_copy['Daily_Return'] = df_copy[price_col].pct_change()
        logger.info(f"Calculated daily returns based on '{price_col}'.")
        return df_copy

    def _add_technical_indicators(self, df: pd.DataFrame, indicators: list = None) -> pd.DataFrame:
        """
        Adds technical indicators to the DataFrame using the pandas_ta library.
        
        Args:
            df (pd.DataFrame): The input DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns.
            indicators (list): A list of indicator specifications. Each item can be:
                               - A string: The name of the indicator (e.g., 'SMA', 'RSI').
                               - A tuple: (indicator_name, params_dict) for parameterized indicators
                                 (e.g., ('SMA', {'length': 20}), ('MACD', {'fast': 12, 'slow': 26, 'signal': 9})).
                               If None, a default set of common indicators will be added.

        Returns:
            pd.DataFrame: DataFrame with added technical indicator columns.
        """
        df_copy = df.copy()
        
        if indicators is None:
            indicators = [
                'SMA', 'RSI', 'MACD', 'BBANDS', 'ADX', 'ATR'
            ]
            logger.info("No specific indicators provided. Adding default set: SMA, RSI, MACD, BBANDS, ADX, ATR.")

        for indicator_spec in indicators:
            indicator_name = None
            indicator_params = {}

            if isinstance(indicator_spec, str):
                indicator_name = indicator_spec.upper()
            elif isinstance(indicator_spec, tuple) and len(indicator_spec) >= 1:
                indicator_name = indicator_spec[0].upper()
                if len(indicator_spec) > 1 and isinstance(indicator_spec[1], dict):
                    indicator_params = indicator_spec[1]
            else:
                logger.warning(f"Invalid indicator specification format: {indicator_spec}. Skipping.")
                continue

            try:
                # pandas_ta functions are called via the .ta accessor.
                # Ensure 'append=True' to add new columns directly to the DataFrame.
                if 'append' not in indicator_params:
                    indicator_params['append'] = True

                # Dynamically get the indicator function from df.ta
                indicator_func = getattr(df_copy.ta, indicator_name.lower(), None)
                
                if indicator_func:
                    indicator_func(**indicator_params)
                    logger.info(f"Added indicator: {indicator_name} with params: {indicator_params}")
                else:
                    logger.warning(f"Indicator '{indicator_name}' not found in pandas_ta or not applicable. Skipping.")

            except Exception as e:
                logger.error(f"Error adding indicator '{indicator_name}': {e}")
                # Continue processing other indicators even if one fails.

        return df_copy

    def prepare_data_for_backtesting(self, df: pd.DataFrame, indicators: list = None) -> pd.DataFrame:
        """
        Orchestrates the complete data cleaning, transformation, and preparation steps
        for backtesting. This is the main entry point for data processing.
        
        The steps are:
        1. Standardize column names.
        2. Clean data (handle missing values, ensure DatetimeIndex, sort, remove duplicates).
        3. Calculate daily returns.
        4. Add specified technical indicators.
        5. Final cleanup by dropping rows with NaNs (often introduced by indicators).

        Args:
            df (pd.DataFrame): The raw historical data DataFrame.
            indicators (list): A list of indicator specifications to add.
                               See _add_technical_indicators for format. If None, default indicators are added.

        Returns:
            pd.DataFrame: The processed DataFrame, ready for strategy backtesting.
        
        Raises:
            ValueError: If the input DataFrame is empty or not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Input DataFrame is empty or not a pandas DataFrame.")
            raise ValueError("Input DataFrame cannot be empty or not a pandas DataFrame.")

        logger.info("Starting data preparation for backtesting...")
        
        # Step 1: Standardize column names
        processed_df = self._standardize_columns(df)
        logger.info("Columns standardized.")

        # Step 2: Clean data (handle missing values, sort index, check duplicates)
        processed_df = self._clean_data(processed_df)
        logger.info("Data cleaned.")

        # Step 3: Calculate daily returns
        if 'Close' in processed_df.columns:
            processed_df = self._calculate_returns(processed_df, price_col='Close')
            logger.info("Daily returns calculated.")
        else:
            logger.warning("Cannot calculate daily returns: 'Close' column not found after cleaning.")

        # Step 4: Add technical indicators
        processed_df = self._add_technical_indicators(processed_df, indicators=indicators)
        logger.info("Technical indicators added.")
        
        # Final cleanup: Drop any rows that might have NaNs introduced by indicators
        # (e.g., first few rows for SMA, MACD which require a lookback period).
        initial_rows_after_indicators = len(processed_df)
        if processed_df.isnull().any().any():
            nan_rows_after_indicators = processed_df.isnull().any(axis=1).sum()
            processed_df.dropna(inplace=True)
            if len(processed_df) < initial_rows_after_indicators:
                logger.warning(f"Dropped {nan_rows_after_indicators} rows due to NaNs after indicator calculation.")

        logger.info(f"Data preparation complete. Final DataFrame shape: {processed_df.shape}")
        return processed_df

# Example usage (for testing this module in isolation)
if __name__ == '__main__':
    # Setup basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create a dummy DataFrame resembling Yahoo Finance data with some issues
    data = {
        'Date': pd.to_datetime([
            '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', 
            '2023-01-06', '2023-01-06', # Duplicate date
            '2023-01-08', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12',
            '2023-01-13', '2023-01-14', '2023-01-15', '2023-01-16', '2023-01-17'
        ]),
        'Open': [np.nan, 101, 102, 103, 104, 105, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116],
        'High': [102, 103, 104, 105, 106, 107, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118],
        'Low': [99, 100, 101, 102, 103, 104, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
        'Close': [101, 102, 103, np.nan, 105, 106, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
        'Adj Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 105.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5],
        'Volume': [100000, 110000, 120000, 130000, 140000, np.nan, 150000, 170000, 180000, 190000, 200000, 210000, 220000, 230000, 240000, 250000, 260000],
        'Dividends': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0] # Extra column
    }

    raw_df = pd.DataFrame(data).set_index('Date')
    
    print("\n--- Raw DataFrame ---")
    print(raw_df)
    print(raw_df.info())
    print(f"Raw DataFrame has {raw_df.isnull().sum().sum()} NaNs.")

    processor = DataProcessor()

    # Test with default indicators
    print("\n--- Processing with default indicators ---")
    processed_df_default = processor.prepare_data_for_backtesting(raw_df.copy())
    print("\n--- Processed DataFrame (Default Indicators) ---")
    print(processed_df_default.head(10)) # Show more rows to see indicator NaNs
    print(processed_df_default.info())
    print(processed_df_default.columns.tolist())
    print(f"Shape: {processed_df_default.shape}")
    print(f"Processed DataFrame has {processed_df_default.isnull().sum().sum()} NaNs.")


    # Test with custom indicators
    print("\n--- Processing with custom indicators ---")
    custom_indicators = [
        ('SMA', {'length': 5}),
        ('RSI', {'length': 7}),
        'MACD',
        ('VWAP', {}) # Example of an indicator that might need specific columns
    ]
    processed_df_custom = processor.prepare_data_for_backtesting(raw_df.copy(), indicators=custom_indicators)
    print("\n--- Processed DataFrame (Custom Indicators) ---")
    print(processed_df_custom.head(10))
    print(processed_df_custom.info())
    print(processed_df_custom.columns.tolist())
    print(f"Shape: {processed_df_custom.shape}")
    print(f"Processed DataFrame has {processed_df_custom.isnull().sum().sum()} NaNs.")

    # Test with an empty DataFrame
    print("\n--- Testing with empty DataFrame ---")
    try:
        processor.prepare_data_for_backtesting(pd.DataFrame())
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Test with a DataFrame that has only some columns
    print("\n--- Testing with partial DataFrame (only Close) ---")
    partial_data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
        'Close': [10, 11, 12, 13, 14]
    }
    partial_df = pd.DataFrame(partial_data).set_index('Date')
    try:
        processed_partial_df = processor.prepare_data_for_backtesting(partial_df)
        print("\n--- Processed Partial DataFrame ---")
        print(processed_partial_df.head())
        print(processed_partial_df.columns.tolist())
        print(f"Shape: {processed_partial_df.shape}")
    except ValueError as e:
        print(f"Caught error for partial DataFrame: {e}")
    except Exception as e:
        print(f"Caught unexpected error for partial DataFrame: {e}")