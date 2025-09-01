import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')


def get_sector_data(start_date='2015-01-01', end_date='2024-12-31', fetch_data=False):
    """
    Fetch S&P 500 sector ETF data
    
    Parameters:
    -----------
    start_date : str
        Start date for data fetching in 'YYYY-MM-DD' format
    end_date : str
        End date for data fetching in 'YYYY-MM-DD' format
    fetch_data : bool
        If True, fetch fresh data from Yahoo Finance. If False, load from cached pickle file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing daily returns for all sector ETFs
    """
    # S&P 500 sector ETFs (SPDRs)
    sector_etfs = {
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLE': 'Energy',
        'XLF': 'Financials',
        'XLV': 'Health Care',
        'XLI': 'Industrials',
        'XLK': 'Technology',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLU': 'Utilities',
        'XLC': 'Communication Services'
    }

    if fetch_data:
        print("Downloading sector ETF data...")
        data = yf.download(list(sector_etfs.keys()), start=start_date, end=end_date)
        data = data.xs('Close', axis=1, level=0)
        
        data.to_pickle('data/data.pkl')
    else:
        data = pd.read_pickle('data/data.pkl')
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    print(f"Data shape: {returns.shape}")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    
    return returns


def get_sector_info():
    """
    Get information about available sector ETFs
    
    Returns:
    --------
    dict
        Dictionary mapping ETF symbols to sector names
    """
    return {
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLE': 'Energy',
        'XLF': 'Financials',
        'XLV': 'Health Care',
        'XLI': 'Industrials',
        'XLK': 'Technology',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLU': 'Utilities',
        'XLC': 'Communication Services'
    }


if __name__ == "__main__":
    # Example usage
    print("Testing data loading module...")
    print("=" * 50)
    
    # Load data from cache
    returns_data = get_sector_data(fetch_data=False)
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(returns_data.describe())
    
    # Display sector information
    print("\nAvailable Sectors:")
    sectors = get_sector_info()
    for symbol, name in sectors.items():
        print(f"  {symbol}: {name}")