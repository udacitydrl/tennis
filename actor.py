import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import helper

class Actor(nn.Module):
    """Actor (Policy) Model."""
    
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
        # setup actor network architecture
        self.input = nn.Linear(state_size, hidden_units[0])
        # apply batch normalization to the output of the first layer
        self.batch_norm = nn.BatchNorm1d(hidden_units[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(size_in, size_out) for size_in, size_out in zip(hidden_units[:-1], hidden_units[1:])])
        self.output = nn.Linear(hidden_units[-1], action_size)
        # initialize the weights and biases of the network
        helper.reset_parameters(self)
        
    def forward(self, state):
        """
        Build the actor network: states -> actions
        Params, state: tuple
        """
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        # apply relu activation to the outputs of all hidden layers
        x = F.relu(self.input(state))
        # in addition, apply batch normalization to the putput of the first layer
        x = self.batch_norm(x)      
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        # the activation tanh is applied to the final output to bound the action
        return F.tanh(self.output(x))