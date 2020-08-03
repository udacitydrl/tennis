import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import helper
    
class Critic(nn.Module):
    """Critic (Value) Model."""
  
    def __init__(self, state_size, action_size, hidden_units, seed):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes of hidden layers
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed) 
        # setup critic
        units = hidden_units.copy()
        # apply batch normalization to the output of the first layer
        self.input = nn.Linear(state_size, units[0])
        self.batch_norm = nn.BatchNorm1d(units[0])
        units[0] += action_size
        self.hidden_layers = nn.ModuleList([nn.Linear(size_in, size_out) for size_in, size_out in zip(units[:-1], units[1:])])
        self.output = nn.Linear(units[-1], 1)
        # initialize the weights and biases of the network
        helper.reset_parameters(self)
        
    def forward(self, state, action):
        """
        Build the actor network: states -> actions
        Params
        ======
            state(tuple)
            actiom(tuple)
        """
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        # apply relu activation to the outputs of all hidden layers
        x0 = F.relu(self.input(state))
        # in addition, apply batch normalization to the putput of the first layer
        x0 = self.batch_norm(x0)
        # include action from the secons layer
        x = torch.cat((x0, action), dim=1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output(x)