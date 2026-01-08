import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RewardNetwork(nn.Module):
    def __init__(self, input_dim, lr=0.01):
        super(RewardNetwork, self).__init__()
        
        # A simple 2-layer neural network
        # Input (4) -> Hidden (64) -> Output (1 scalar reward)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )
        
        # We use Adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        """Standard forward pass"""
        return self.model(state)

    def predict_reward(self, state_np):
        """
        Helper for the Environment Wrapper.
        Takes a numpy array state, returns a python float reward.
        """
        # Convert numpy state to torch tensor
        state_t = torch.FloatTensor(state_np).unsqueeze(0) # Add batch dim
        
        with torch.no_grad():
            reward = self.forward(state_t)
            
        return reward.item() # Return standard float

    def train_on_batch(self, traj_A_states, traj_B_states, label_A_is_better):
        """
        The Magic: Updates weights based on preference.
        
        traj_A_states: Numpy array of states in Trajectory A
        traj_B_states: Numpy array of states in Trajectory B
        label_A_is_better: 1 if A > B, 0 if B > A
        """
        # 1. Prepare data
        states_A = torch.FloatTensor(traj_A_states)
        states_B = torch.FloatTensor(traj_B_states)
        
        # 2. Reset gradients
        self.optimizer.zero_grad()
        
        # 3. Predict rewards for every step in both trajectories
        # (This uses the current weights of the network)
        r_A = self.forward(states_A)
        r_B = self.forward(states_B)
        
        # 4. Sum them up to get "Return" (Total Reward)
        sum_r_A = torch.sum(r_A)
        sum_r_B = torch.sum(r_B)
        
        # 5. Calculate Loss (Bradley-Terry Model)
        # Probability that A is better = Sigmoid(Sum_A - Sum_B)
        # If label is 1 (A better), we want (Sum_A - Sum_B) to be big positive.
        # If label is 0 (B better), we want (Sum_A - Sum_B) to be big negative.
        
        diff = sum_r_A - sum_r_B
        
        # We use Binary Cross Entropy with Logits
        # This combines Sigmoid + Log Loss
        criterion = nn.BCEWithLogitsLoss()
        
        # Create the target tensor (1.0 or 0.0)
        target = torch.tensor([label_A_is_better], dtype=torch.float32)
        
        # Calculate loss between our difference and the target
        loss = criterion(diff.unsqueeze(0), target)
        
        # 6. Backpropagate
        loss.backward()
        self.optimizer.step()
        
        return loss.item()