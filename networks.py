import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    """
    Two-layer MLP policy network that outputs portfolio weights
    """

    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mean = self.mean(x)

        # Apply softmax to ensure weights sum to 1 and are positive
        weights = F.softmax(mean, dim=-1)
        std = torch.exp(self.log_std).expand_as(weights)

        return Normal(weights, std)


class ValueNetwork(nn.Module):
    """
    Critic network for advantage estimation
    """

    def __init__(self, state_dim, hidden_size=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return self.value(x)
