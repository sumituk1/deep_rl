import numpy as np


class DifferentialSharpeCalculator:
    """
    Calculates differential Sharpe ratio for immediate rewards
    
    The differential Sharpe ratio provides an instantaneous measure of how
    a new return affects the overall Sharpe ratio, used as a reward signal
    in reinforcement learning for portfolio optimization.
    """

    def __init__(self):
        self.returns = []
        self.mean_return = 0
        self.variance_return = 0
        self.n = 0

    def update(self, new_return):
        """
        Update with new return and calculate differential Sharpe
        
        Parameters:
        -----------
        new_return : float
            The portfolio return for the current period
            
        Returns:
        --------
        float
            The differential Sharpe ratio (reward signal)
        """
        self.returns.append(new_return)
        self.n += 1

        if self.n == 1:
            self.mean_return = new_return
            self.variance_return = 0
            return 0

        # Update running statistics using Welford's online algorithm
        old_mean = self.mean_return
        self.mean_return = old_mean + (new_return - old_mean) / self.n
        self.variance_return = (
            ((self.n - 2) * self.variance_return + (new_return - old_mean) * (new_return - self.mean_return)) /
            (self.n - 1)
        )

        if self.variance_return <= 0:
            return 0

        # Calculate differential Sharpe ratio
        std_return = np.sqrt(self.variance_return)
        sharpe = self.mean_return / std_return

        # Differential Sharpe approximation using Taylor expansion
        delta_sharpe = (
            (new_return - self.mean_return) / (std_return ** 2) -
            0.5 * sharpe * (new_return - self.mean_return) ** 2 /
            (std_return ** 3)
        )

        return delta_sharpe

    def reset(self):
        """Reset the calculator to initial state"""
        self.returns = []
        self.mean_return = 0
        self.variance_return = 0
        self.n = 0


def calculate_sharpe_ratio(returns, annualize=True):
    """
    Calculate Sharpe ratio from returns
    
    Parameters:
    -----------
    returns : array-like
        Array of returns
    annualize : bool
        Whether to annualize the Sharpe ratio (assumes daily returns)
        
    Returns:
    --------
    float
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0
    
    returns_array = np.array(returns)
    sharpe = np.mean(returns_array) / (np.std(returns_array) + 1e-8)
    
    if annualize:
        sharpe *= np.sqrt(252)  # Assuming daily returns
    
    return sharpe


def benchmark_comparison(returns_data):
    """
    Calculate benchmark portfolio performance
    
    Parameters:
    -----------
    returns_data : pd.DataFrame
        DataFrame containing asset returns
        
    Returns:
    --------
    tuple
        (equal_weight_sharpe, market_weight_sharpe)
    """
    print("\nCalculating benchmark performance...")
    
    # Equal weight benchmark
    ew_returns = returns_data.mean(axis=1)
    ew_sharpe = calculate_sharpe_ratio(ew_returns)
    
    # Market cap weight approximation (XLK technology having higher weight)
    market_weights = np.array([0.15, 0.08, 0.04, 0.13, 0.14, 0.08, 0.25, 0.03, 0.03, 0.04, 0.03])
    market_weights = market_weights / market_weights.sum()
    mw_returns = (returns_data * market_weights).sum(axis=1)
    mw_sharpe = calculate_sharpe_ratio(mw_returns)
    
    print(f"Equal Weight Sharpe: {ew_sharpe:.4f}")
    print(f"Market Weight Sharpe: {mw_sharpe:.4f}")
    
    return ew_sharpe, mw_sharpe