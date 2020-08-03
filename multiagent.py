from agent import DDPG, device
import numpy as np

import torch
import random
from collections import namedtuple, deque

class MultiDDPG:
    """multiple DDPG agents"""
    
    def __init__(self, num_agents, state_size, action_size, params, seed):
        """Initialize multiple DDPG agents
        
        Params
        ======
            num_agents (int): number of DDPG agents
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            params (Params): hyperparameters 
            seed (int): random seed
        """
        self.agents = [DDPG(state_size, action_size, params, seed) for _ in range(num_agents)]
        
        # Replay Buffer forall agents
        self.memory = ReplayBuffer(params.buffer_size, params.batch_size, seed)
        
    def noise_reset(self):
        for agent in self.agents:
            agent.noise_reset()
            
    def act(self, states):
        return [agent.act(state) for agent, state in zip(self.agents, states)]
        

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay buffer, and use random sample from buffer to learn."""
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        
        # learn if enough sample in memory
        if len(self.memory) > self.memory.batch_size:
            for agent in self.agents:
                experiences = self.memory.sample()
                agent.learn(experiences)
                
                
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
            seed (int): random seed
        """
        self.seed = np.random.seed(seed)
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        """ convert to states, actions, rewards, next_states and dones of the selected experiences to tensors """
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return  (states, actions, rewards, next_states, dones)
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
        