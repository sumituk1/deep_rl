import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from networks import PolicyNetwork, ValueNetwork


class PPOAgent:
    """
    Proximal Policy Optimization Agent for Portfolio Management
    """

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, gae_lambda=0.95):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.critic = ValueNetwork(state_dim)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def get_action(self, state):
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            value = self.critic(state_tensor)

        dist = self.policy(state_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        # Ensure weights sum to 1 (should already be true due to softmax)
        action = action / action.sum(-1, keepdim=True)

        return action.squeeze().numpy(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store experience for training"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, next_value=0):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        advantage = 0

        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * advantage
            advantages.insert(0, advantage)

        return advantages

    def update(self):
        """Update policy and value networks"""
        if len(self.states) == 0:
            return

        # Compute advantages
        advantages = self.compute_gae()

        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(self.log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(self.values)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(10):  # Multiple epochs
            # Policy update
            dist = self.policy(states)
            new_log_probs = dist.log_prob(actions).sum(-1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Value update
            values = self.critic(states).squeeze()
            value_loss = F.mse_loss(values, returns)

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        # Clear memory
        self.clear_memory()

    def clear_memory(self):
        """Clear stored experiences"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []