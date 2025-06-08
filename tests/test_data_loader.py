import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import configparser
from datetime import datetime
import logging

# Import the DataLoader class from the source module
from src.data.data_loader import DataLoader

# Define a fixture for a common configuration file
@pytest.fixture
def mock_config_path():
    """Creates a temporary config.ini file for testing.
    This config is used as a base and can be modified by tests.
    """
    config_content = """
[DATA]
data_source = yfinance
ticker = MSFT
start_date = 2023-01-01
end_date = 2023-01-31
local_csv_path = data/mock_data.csv
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ini') as f:
        f.write(config_content)
        temp_path = f.name
    yield temp_path
    os.remove(temp_path)

@pytest.fixture
def mock_yfinance_data():
    """Returns a sample pandas DataFrame mimicking yfinance output structure."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='D'))
    data = {
        'Open': np.random.rand(10) * 100,
        'High': np.random.rand(10) * 100 + 10,
        'Low': np.random.rand(10) * 100 - 10,
        'Close': np.random.rand(10) * 100,
        'Adj Close': np.random.rand(10) * 100,
        'Volume': np.random.randint(100000, 1000000, 10)
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

@pytest.fixture
def mock_local_csv_data():
    """Returns a sample pandas DataFrame for local CSV testing.
    Includes a 'Date' column as string, typical for CSVs before parsing.
    """
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='D'))
    data = {
        'Date': dates.strftime('%Y-%m-%d'), # CSV often has date as string
        'Open': np.random.rand(10) * 100,
        'High': np.random.rand(10) * 100 + 10,
        'Low': np.random.rand(10) * 100 - 10,
        'Close': np.random.rand(10) * 100,
        'Adj Close': np.random.rand(10) * 100,
        'Volume': np.random.randint(100000, 1000000, 10)
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def temp_csv_file(mock_local_csv_data):
    """Creates a temporary CSV file with mock data for local_csv tests."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        mock_local_csv_data.to_csv(f.name, index=False)
        temp_path = f.name
    yield temp_path
    os.remove(temp_path)

class TestDataLoader:
    """
    Unit tests for the DataLoader class in src/data/data_loader.py.
    """

    def test_initialization_and_config_loading(self, mock_config_path):
        """Test if DataLoader initializes and loads config correctly from a valid path."""
        loader = DataLoader(config_path=mock_config_path)
        assert loader.config is not None
        assert loader.config['DATA']['data_source'] == 'yfinance'
        assert loader.config['DATA']['start_date'] == '2023-01-01'
        assert loader.config['DATA']['end_date'] == '2023-01-31'
        assert loader.config['DATA']['ticker'] == 'MSFT'

    def test_initialization_with_non_existent_config(self):
        """Test DataLoader initialization with a non-existent config path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            DataLoader(config_path="non_existent_config.ini")

    @patch('yfinance.download')
    def test_load_data_yfinance(self, mock_yf_download, mock_config_path, mock_yfinance_data):
        """Test successful data loading from yfinance."""
        mock_yf_download.return_value = mock_yfinance_data

        loader = DataLoader(config_path=mock_config_path)
        # The config fixture already sets yfinance as data_source and provides ticker/dates

        df = loader.load_data()

        # Assert yfinance.download was called with correct parameters
        mock_yf_download.assert_called_once_with(
            loader.config['DATA']['ticker'],
            start=pd.to_datetime(loader.config['DATA']['start_date']),
            end=pd.to_datetime(loader.config['DATA']['end_date']),
            progress=False
        )
        # Assert the returned DataFrame matches the mock data
        pd.testing.assert_frame_equal(df, mock_yfinance_data)
        assert df.index.name == 'Date'
        assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

    @patch('yfinance.download')
    def test_load_data_yfinance_no_data(self, mock_yf_download, mock_config_path, caplog):
        """Test yfinance data loading when no data is returned (e.g., invalid ticker/dates)."""
        mock_yf_download.return_value = pd.DataFrame() # Simulate no data returned

        loader = DataLoader(config_path=mock_config_path)
        loader.config['DATA']['data_source'] = 'yfinance'
        loader.config['DATA']['ticker'] = 'NONEXISTENT'
        loader.config['DATA']['start_date'] = '2023-01-01'
        loader.config['DATA']['end_date'] = '2023-01-02'

        with caplog.at_level(logging.WARNING):
            df = loader.load_data()
            assert df.empty
            assert "No data downloaded for ticker 'NONEXISTENT' from yfinance" in caplog.text

    @patch('pandas.read_csv')
    def test_load_data_local_csv(self, mock_read_csv, mock_config_path, mock_local_csv_data, temp_csv_file):
        """Test successful data loading from a local CSV file."""
        # pandas.read_csv will return the raw mock_local_csv_data (with 'Date' as a column)
        mock_read_csv.return_value = mock_local_csv_data.copy()

        loader = DataLoader(config_path=mock_config_path)
        loader.config['DATA']['data_source'] = 'local_csv'
        loader.config['DATA']['local_csv_path'] = temp_csv_file # Use the actual temp file path
        loader.config['DATA']['start_date'] = '2023-01-01'
        loader.config['DATA']['end_date'] = '2023-01-10'

        df = loader.load_data()

        # Assert pandas.read_csv was called with correct parameters
        mock_read_csv.assert_called_once_with(
            temp_csv_file,
            index_col='Date',
            parse_dates=True
        )
        # The DataLoader's _load_local_csv method handles date filtering and validation.
        # We need to manually prepare the expected DataFrame after parsing and filtering
        expected_df = mock_local_csv_data.set_index(pd.to_datetime(mock_local_csv_data['Date']))
        expected_df.index.name = 'Date'
        expected_df = expected_df.drop(columns=['Date']) # Drop the original 'Date' column as it's now the index

        # Apply date filtering as DataLoader would
        start_date_filter = pd.to_datetime(loader.config['DATA']['start_date'])
        end_date_filter = pd.to_datetime(loader.config['DATA']['end_date'])
        expected_df = expected_df.loc[
            (expected_df.index >= start_date_filter) &
            (expected_df.index <= end_date_filter)
        ]

        pd.testing.assert_frame_equal(df, expected_df)
        assert df.index.name == 'Date'
        assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

    def test_load_data_local_csv_missing_file(self, mock_config_path, caplog):
        """Test loading data from a non-existent local CSV file raises FileNotFoundError."""
        loader = DataLoader(config_path=mock_config_path)
        loader.config['DATA']['data_source'] = 'local_csv'
        loader.config['DATA']['local_csv_path'] = 'non_existent_data.csv'

        with caplog.at_level(logging.ERROR):
            with pytest.raises(FileNotFoundError, match="CSV file not found at 'non_existent_data.csv'"):
                loader.load_data()
            assert "CSV file not found at 'non_existent_data.csv'" in caplog.text

    def test_load_data_invalid_source(self, mock_config_path, caplog):
        """Test loading data with an unsupported data source raises ValueError."""
        loader = DataLoader(config_path=mock_config_path)
        loader.config['DATA']['data_source'] = 'unsupported_source'

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="Unsupported data source: unsupported_source"):
                loader.load_data()
            assert "Unsupported data source: unsupported_source" in caplog.text

    def test_validate_dataframe_success(self, mock_config_path, mock_yfinance_data):
        """Test _validate_dataframe with a valid DataFrame (all required columns, proper index)."""
        loader = DataLoader(config_path=mock_config_path)
        # No exception should be raised
        validated_df = loader._validate_dataframe(mock_yfinance_data)
        pd.testing.assert_frame_equal(validated_df, mock_yfinance_data)


    def test_validate_dataframe_missing_column(self, mock_config_path, mock_yfinance_data, caplog):
        """Test _validate_dataframe with a DataFrame missing a required column."""
        df_missing_col = mock_yfinance_data.drop(columns=['Close'])
        loader = DataLoader(config_path=mock_config_path)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="DataFrame is missing required columns: {'Close'}"):
                loader._validate_dataframe(df_missing_col)
            assert "DataFrame is missing required columns" in caplog.text
            assert "'Close'" in caplog.text

    def test_validate_dataframe_empty(self, mock_config_path, caplog):
        """Test _validate_dataframe with an empty DataFrame."""
        loader = DataLoader(config_path=mock_config_path)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="No data loaded or DataFrame is empty."):
                loader._validate_dataframe(pd.DataFrame())
            assert "No data loaded or DataFrame is empty." in caplog.text

    def test_validate_dataframe_index_name(self, mock_config_path, mock_yfinance_data, caplog):
        """Test _validate_dataframe ensures index is named 'Date' and logs a warning if not."""
        df_no_index_name = mock_yfinance_data.copy()
        df_no_index_name.index.name = None # Remove index name

        loader = DataLoader(config_path=mock_config_path)

        with caplog.at_level(logging.WARNING):
            validated_df = loader._validate_dataframe(df_no_index_name)
            assert validated_df.index.name == 'Date'
            assert "DataFrame index is not named 'Date'. Renaming it." in caplog.text
            # Check that the data itself remains the same (excluding index name)
            pd.testing.assert_frame_equal(validated_df.reset_index(drop=True), df_no_index_name.reset_index(drop=True))

    def test_date_filtering_yfinance(self, mock_config_path, mock_yfinance_data):
        """Test date filtering logic for yfinance data."""
        start_date_str = '2023-01-03'
        end_date_str = '2023-01-07'

        loader = DataLoader(config_path=mock_config_path)
        loader.config['DATA']['start_date'] = start_date_str
        loader.config['DATA']['end_date'] = end_date_str
        loader.config['DATA']['ticker'] = 'TEST' # Required for yfinance path

        with patch('yfinance.download', return_value=mock_yfinance_data) as mock_yf_download:
            df = loader.load_data()

            expected_start = pd.to_datetime(start_date_str)
            expected_end = pd.to_datetime(end_date_str)

            assert not df.empty
            assert df.index.min() >= expected_start
            assert df.index.max() <= expected_end
            assert len(df) == 5 # 2023-01-03 to 2023-01-07 (inclusive)

    def test_date_filtering_local_csv(self, mock_config_path, mock_local_csv_data, temp_csv_file):
        """Test date filtering logic for local CSV data."""
        start_date_str = '2023-01-03'
        end_date_str = '2023-01-07'

        loader = DataLoader(config_path=mock_config_path)
        loader.config['DATA']['data_source'] = 'local_csv'
        loader.config['DATA']['local_csv_path'] = temp_csv_file
        loader.config['DATA']['start_date'] = start_date_str
        loader.config['DATA']['end_date'] = end_date_str

        df = loader.load_data()

        expected_start = pd.to_datetime(start_date_str)
        expected_end = pd.to_datetime(end_date_str)

        assert not df.empty
        assert df.index.min() >= expected_start
        assert df.index.max() <= expected_end
        assert len(df) == 5 # 2023-01-03 to 2023-01-07 (inclusive)

    def test_load_data_config_no_dates(self, mock_config_path, mock_yfinance_data):
        """Test data loading when start/end dates are not specified in config."""
        # Modify config to remove dates
        config = configparser.ConfigParser()
        config.read(mock_config_path)
        if 'start_date' in config['DATA']:
            del config['DATA']['start_date']
        if 'end_date' in config['DATA']:
            del config['DATA']['end_date']

        # Write modified config to a new temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ini') as f:
            config.write(f)
            temp_config_path = f.name
        
        loader = DataLoader(config_path=temp_config_path)
        loader.config['DATA']['ticker'] = 'MSFT' # Ensure ticker is set for yfinance path

        with patch('yfinance.download', return_value=mock_yfinance_data) as mock_yf_download:
            df = loader.load_data()
            # yfinance.download should be called without start/end dates (i.e., None)
            mock_yf_download.assert_called_once_with(
                'MSFT',
                start=None,
                end=None,
                progress=False
            )
            pd.testing.assert_frame_equal(df, mock_yfinance_data)
        
        os.remove(temp_config_path) # Clean up temp config

    def test_load_data_config_invalid_dates(self, mock_config_path, caplog):
        """Test data loading with invalid date formats in config logs warnings and sets dates to None."""
        config = configparser.ConfigParser()
        config.read(mock_config_path)
        config['DATA']['start_date'] = 'invalid-date'
        config['DATA']['end_date'] = 'not-a-date' # Both invalid

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ini') as f:
            config.write(f)
            temp_config_path = f.name

        loader = DataLoader(config_path=temp_config_path)
        loader.config['DATA']['ticker'] = 'MSFT' # Ensure ticker is set for yfinance path

        with patch('yfinance.download', return_value=pd.DataFrame()) as mock_yf_download:
            with caplog.at_level(logging.WARNING):
                df = loader.load_data()
                mock_yf_download.assert_called_once_with(
                    'MSFT',
                    start=None, # Should be None due to invalid format
                    end=None,   # Should be None due to invalid format
                    progress=False
                )
                assert "Could not parse start_date 'invalid-date'" in caplog.text
                assert "Could not parse end_date 'not-a-date'" in caplog.text
                assert df.empty # Assuming yfinance.download returns empty df for this test

        os.remove(temp_config_path)

    def test_load_data_yfinance_missing_ticker_in_config(self, mock_config_path, caplog):
        """Test yfinance data loading raises ValueError if ticker is not specified in config."""
        loader = DataLoader(config_path=mock_config_path)
        loader.config['DATA']['data_source'] = 'yfinance'
        if 'ticker' in loader.config['DATA']:
            del loader.config['DATA']['ticker'] # Ensure ticker is missing

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="Ticker symbol not specified in config for yfinance data source."):
                loader.load_data()
            assert "Ticker symbol not specified in config for yfinance data source." in caplog.text

    def test_load_data_local_csv_missing_path_in_config(self, mock_config_path, caplog):
        """Test local CSV data loading raises ValueError if path is not specified in config."""
        loader = DataLoader(config_path=mock_config_path)
        loader.config['DATA']['data_source'] = 'local_csv'
        if 'local_csv_path' in loader.config['DATA']:
            del loader.config['DATA']['local_csv_path']

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="Local CSV path not specified in config for local_csv data source."):
                loader.load_data()
            assert "Local CSV path not specified in config for local_csv data source." in caplog.text