"""
Tests for GitHub data fetcher module.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import pandas as pd

from src.ingestion.github_data import (
    GitHubDataFetcher,
    fetch_fivethirtyeight_elo,
    FIVETHIRTYEIGHT_URLS,
)


@pytest.mark.asyncio
async def test_fetch_csv():
    """Test fetching CSV from GitHub."""
    fetcher = GitHubDataFetcher()
    
    # Mock CSV content
    csv_content = b"col1,col2\nval1,val2\nval3,val4"
    
    with patch.object(fetcher, "_fetch_raw", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = csv_content
        
        df = await fetcher.fetch_csv("https://raw.githubusercontent.com/test/repo/file.csv")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["col1", "col2"]


@pytest.mark.asyncio
async def test_fetch_json():
    """Test fetching JSON from GitHub."""
    fetcher = GitHubDataFetcher()
    
    # Mock JSON content
    json_content = b'{"key": "value", "number": 42}'
    
    with patch.object(fetcher, "_fetch_raw", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = json_content
        
        data = await fetcher.fetch_json("https://raw.githubusercontent.com/test/repo/file.json")
        
        assert isinstance(data, dict)
        assert data["key"] == "value"
        assert data["number"] == 42


@pytest.mark.asyncio
async def test_fetch_text():
    """Test fetching text from GitHub."""
    fetcher = GitHubDataFetcher()
    
    # Mock text content
    text_content = b"Hello, World!\nThis is a test file."
    
    with patch.object(fetcher, "_fetch_raw", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = text_content
        
        text = await fetcher.fetch_text("https://raw.githubusercontent.com/test/repo/file.txt")
        
        assert isinstance(text, str)
        assert "Hello, World!" in text


@pytest.mark.asyncio
async def test_fetch_auto_detect_format():
    """Test automatic format detection."""
    fetcher = GitHubDataFetcher()
    
    csv_content = b"col1,col2\nval1,val2"
    
    with patch.object(fetcher, "_fetch_raw", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = csv_content
        
        # Auto-detect CSV from .csv extension
        result = await fetcher.fetch("https://raw.githubusercontent.com/test/repo/file.csv")
        
        assert result.format == "csv"
        assert isinstance(result.data, pd.DataFrame)


def test_fivethirtyeight_urls_exist():
    """Test that FiveThirtyEight URLs are defined."""
    assert "elo_historical" in FIVETHIRTYEIGHT_URLS
    assert "elo_latest" in FIVETHIRTYEIGHT_URLS
    assert "fivethirtyeight" in FIVETHIRTYEIGHT_URLS["elo_historical"].lower()


@pytest.mark.asyncio
async def test_fetch_fivethirtyeight_elo_invalid_dataset():
    """Test that invalid dataset name raises KeyError."""
    with pytest.raises(KeyError):
        await fetch_fivethirtyeight_elo("invalid_dataset_name")


@pytest.mark.asyncio
@patch("src.ingestion.github_data.GitHubDataFetcher")
async def test_fetch_fivethirtyeight_elo_success(mock_fetcher_class):
    """Test successful fetch of FiveThirtyEight ELO data."""
    # Mock the fetcher instance
    mock_fetcher = MagicMock()
    mock_fetcher_class.return_value = mock_fetcher
    
    # Mock the fetch_csv method
    mock_df = pd.DataFrame({"col1": [1, 2, 3]})
    mock_fetcher.fetch_csv = AsyncMock(return_value=mock_df)
    
    # Replace the fetcher in the function
    import src.ingestion.github_data as github_module
    original_fetcher = github_module.GitHubDataFetcher
    github_module.GitHubDataFetcher = mock_fetcher_class
    
    try:
        df = await fetch_fivethirtyeight_elo("elo_historical")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
    finally:
        github_module.GitHubDataFetcher = original_fetcher
