"""
helper contains a couple of u to reset parameters for the actor and critic networks
"""

import numpy as np

def hidden_init(layer):
    """
    Given a layer, the function returns an interval
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. /np.sqrt(fan_in)
    return (-lim, lim)

def reset_parameters(networks):
    """
    Reset the weights and biases of the given networks, which can be either an actor or a critic
    """
    networks.input.weight.data.uniform_(*hidden_init(networks.input))
    for layer in networks.hidden_layers:
        layer.weight.data.uniform_(*hidden_init(layer))
    # initialize the weights and biases of the final layer by the uniform distribution [−3 × 10−3, 3 × 10−3]
    networks.output.weight.data.uniform_(-3e-3, 3e-3)