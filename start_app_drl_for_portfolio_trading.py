# https://gatambook.substack.com/p/deep-reinforcement-learning-for-portfolio?utm_source=substack&utm_medium=email

import numpy as np
import torch
import warnings

# Import from refactored modules
from load_data import get_sector_data
from portfolio_env import PortfolioEnvironment
from ppo_agent import PPOAgent
from utils import DifferentialSharpeCalculator, calculate_sharpe_ratio, benchmark_comparison

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
FETCH_DATA = False


def validate_agent(agent, val_data):
    """Validate agent on validation data"""
    env = PortfolioEnvironment(val_data)
    state = env.reset()
    
    portfolio_returns = []
    
    for step in range(env.max_steps):
        action, _, _ = agent.get_action(state)
        next_state, portfolio_return, done = env.step(action)
        portfolio_returns.append(portfolio_return)
        
        if done:
            break
        state = next_state
    
    # Calculate Sharpe ratio
    return calculate_sharpe_ratio(portfolio_returns)


def out_of_sample_test_agent(agent, test_data):
    """Test agent on out-of-sample data"""
    env = PortfolioEnvironment(test_data)
    state = env.reset()
    
    portfolio_returns = []
    
    for step in range(env.max_steps):
        action, _, _ = agent.get_action(state)
        next_state, portfolio_return, done = env.step(action)
        portfolio_returns.append(portfolio_return)
        
        if done:
            break
        state = next_state
    
    # Calculate Sharpe ratio
    sharpe = calculate_sharpe_ratio(portfolio_returns)
    return sharpe, portfolio_returns


def train_drl_agent(returns_data, train_years=3, validation_months=3):
    """
    Train DRL agent with rolling windows
    
    Parameters:
    -----------
    returns_data : pd.DataFrame
        Historical returns data
    train_years : int
        Number of years for training window
    validation_months : int
        Number of months for validation window
        
    Returns:
    --------
    list
        List of results for each training window
    """
    print("Training DRL agent...")
    
    # Split data into training periods
    train_days = train_years * 252  # Approximate trading days per year
    val_days = validation_months * 21  # Approximate trading days per month
    
    results = []
    
    # Rolling window training
    start_idx = 0
    while start_idx + train_days + val_days < len(returns_data):
        print(f"\nTraining window: {start_idx} to {start_idx + train_days}")
        
        # Training data
        train_data = returns_data.iloc[start_idx: start_idx + train_days]
        
        # Validation data
        val_data = returns_data.iloc[start_idx + train_days: start_idx + train_days + val_days]
        
        # Test data (next quarter for out-of-sample)
        test_start = start_idx + train_days + val_days
        test_end = min(test_start + 63, len(returns_data))  # ~1 quarter
        test_data = returns_data.iloc[test_start:test_end]
        
        if len(test_data) < 20:  # Need minimum test period
            break
        
        # Create environment
        env = PortfolioEnvironment(train_data)
        
        # Initialize agent
        state_dim = env.state_dim
        action_dim = env.action_dim
        agent = PPOAgent(state_dim, action_dim)
        
        # Train for multiple episodes
        n_episodes = 200
        best_val_sharpe = -np.inf
        best_agent_state = None
        
        for episode in range(n_episodes):  # Repeat the below steps 200 (# episodes) times
            state = env.reset()
            sharpe_calc = DifferentialSharpeCalculator()
            episode_rewards = []
            
            for step in range(env.max_steps):  # for each time step, compute the state/action/reward/value: SAR)
                action, log_prob, value = agent.get_action(state)
                next_state, portfolio_return, done = env.step(action)  # Note: portfolio_return is the t+1 return as a results of executing \pi(a_{t}|s_{t})
                
                # Calculate differential Sharpe as reward (this is ground truth reward).
                # We will use this to train the value network
                reward = sharpe_calc.update(portfolio_return)  # portfolio return(t+1) =\sum_i action(i, t) * returns(i, t+1)
                episode_rewards.append(reward)
                
                agent.store_transition(state, action, reward, log_prob, value, done)
                
                if done:
                    break
                
                state = next_state
            
            # Update agent
            if (episode > 0) & (episode % 10 == 0):  # Update every 10 episodes
                agent.update()
            
            # Validate every 50 episodes
            if episode % 50 == 0 and episode > 0:
                val_sharpe = validate_agent(agent, val_data)
                print(f"Episode {episode}, Validation Sharpe: {val_sharpe:.4f}")
                
                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    best_agent_state = {
                        'policy': agent.policy.state_dict(),
                        'critic': agent.critic.state_dict()
                    }
        
        # Load best agent and test
        if best_agent_state is not None:
            agent.policy.load_state_dict(best_agent_state['policy'])
            agent.critic.load_state_dict(best_agent_state['critic'])
        
        # Test on out-of-sample data
        test_sharpe, test_returns = out_of_sample_test_agent(agent, test_data)
        
        results.append({
            'train_start': start_idx,
            'train_end': start_idx + train_days,
            'test_start': test_start,
            'test_end': test_end,
            'val_sharpe': best_val_sharpe,
            'test_sharpe': test_sharpe,
            'test_returns': test_returns
        })
        
        print(f"Out-of-sample Sharpe ratio: {test_sharpe:.4f}")
        
        # Move window forward by 1 quarter
        start_idx += 63  # ~1 quarter
    
    return results


def run_main():
    """Main execution function"""
    print("DRL Portfolio Optimization Implementation")
    print("=" * 50)
    
    # Load data
    returns_data = get_sector_data(fetch_data=FETCH_DATA)
    
    # Calculate benchmarks
    ew_sharpe, mw_sharpe = benchmark_comparison(returns_data)
    
    # Train DRL agent
    results = train_drl_agent(returns_data)
    
    # Analyze results
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    test_sharpes = [r['test_sharpe'] for r in results if r['test_sharpe'] is not None]
    
    if test_sharpes:
        avg_drl_sharpe = np.mean(test_sharpes)
        std_drl_sharpe = np.std(test_sharpes)
        
        print(f"DRL Average Sharpe: {avg_drl_sharpe:.4f} ± {std_drl_sharpe:.4f}")
        print(f"Equal Weight Sharpe: {ew_sharpe:.4f}")
        print(f"Market Weight Sharpe: {mw_sharpe:.4f}")
        
        print(f"\nDRL vs Equal Weight: {avg_drl_sharpe - ew_sharpe:.4f}")
        print(f"DRL vs Market Weight: {avg_drl_sharpe - mw_sharpe:.4f}")
        
        print(f"\nNumber of test periods: {len(test_sharpes)}")
        print(f"DRL wins vs Equal Weight: {sum(1 for s in test_sharpes if s > ew_sharpe)} / {len(test_sharpes)}")
        print(f"DRL wins vs Market Weight: {sum(1 for s in test_sharpes if s > mw_sharpe)} / {len(test_sharpes)}")
    
    return results, returns_data


if __name__ == "__main__":
    results, data = run_main()
