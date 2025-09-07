import numpy as np
from sklearn.preprocessing import StandardScaler


class PortfolioEnvironment:
    """
    Portfolio management environment for reinforcement learning
    
    This environment simulates portfolio management where an agent
    allocates weights across different assets and cash.
    """

    def __init__(self, returns_data, lookback_window=60):
        """
        Initialize the portfolio environment
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            DataFrame containing historical returns for each asset
        lookback_window : int
            Number of historical days to use as features
        """
        self.returns_data = returns_data
        self.lookback_window = lookback_window
        self.current_step = 0
        self.max_steps = len(returns_data) - lookback_window - 1
        self.n_assets = returns_data.shape[1]
        
        # Initialize portfolio weights (equal weight + cash)
        self.current_weights = np.ones(self.n_assets + 1) / (self.n_assets + 1)  # +1 for cash
        self.weights_history = []
        
        self.scaler = StandardScaler()
        self._prepare_features()

    def _prepare_features(self):
        """Prepare feature matrix from historical data"""
        features = []
        
        for i in range(self.lookback_window, len(self.returns_data)):
            # Historical returns (lookback_window days)
            hist_returns = self.returns_data.iloc[i - self.lookback_window: i].values.flatten()  # 60 (lookback) * 11 features = (660,)
            
            # Current weights
            weights = self.weights_history if i < len(self.weights_history) else self.current_weights
            
            # Simple moving averages
            ma_5 = self.returns_data.iloc[i - 5:i].mean().values
            ma_20 = self.returns_data.iloc[i - 20:i].mean().values if i >= 20 else ma_5
            
            # Volatility features  
            vol_5 = self.returns_data.iloc[i - 5:i].std().values
            vol_20 = self.returns_data.iloc[i - 20:i].std().values if i >= 20 else vol_5
            
            feature_vector = np.concatenate([hist_returns, weights, ma_5, ma_20, vol_5, vol_20])
            features.append(feature_vector)
        
        self.features = np.array(features)
        self.features = self.scaler.fit_transform(self.features)

    def reset(self):
        """
        Reset environment to initial state
        
        Returns:
        --------
        np.array
            Initial state observation
        """
        self.current_step = 0
        self.current_weights = np.ones(self.n_assets + 1) / (self.n_assets + 1)
        return self.features[0]

    def step(self, action):
        """
        Execute one step in the environment
        
        Parameters:
        -----------
        action : np.array
            Portfolio weights for next period
            
        Returns:
        --------
        tuple
            (next_state, reward, done)
            - next_state: Next observation or None if episode ended
            - reward: Portfolio return for the period
            - done: Whether episode has ended
        """
        # Action is the new portfolio weights
        self.current_weights = action
        
        # Calculate returns for next period
        next_returns = self.returns_data.iloc[self.current_step + self.lookback_window]
        
        # Portfolio return (excluding cash, which has 0 return) : R_p(t) = \sum_{i} A(i,t) * r(i, t+1)
        portfolio_return = np.sum(action[:-1] * next_returns.values)  # Exclude cash weight
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        if done:
            next_state = None
        else:
            next_state = self.features[self.current_step]
        
        return next_state, portfolio_return, done

    @property
    def state_dim(self):
        """Get the dimension of the state space"""
        return self.features.shape[1]
    
    @property
    def action_dim(self):
        """Get the dimension of the action space"""
        return self.n_assets + 1  # +1 for cash