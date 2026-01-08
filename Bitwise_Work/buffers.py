import numpy as np
import random

class TrajectoryBuffer:
    def __init__(self):
        # Master list of all completed trajectories
        # Format: [ [obs1, obs2...], [obs1, obs2...] ]
        self.trajectories = []
        
        # Temporary storage for the ongoing episode
        self.current_trajectory = []

    def add_step(self, obs):
        """
        Called after every step in the environment.
        We only need observations for the Reward Model.
        """
        self.current_trajectory.append(obs)

    def commit_trajectory(self):
        """
        Called when an episode ends (done=True).
        Moves the current data to the master list.
        """
        if len(self.current_trajectory) > 0:
            # Save as a numpy array for easier handling later
            self.trajectories.append(np.array(self.current_trajectory))
            self.current_trajectory = []

    def sample_pair(self):
        """
        Pick two random trajectories for the Critic to compare.
        Returns: (traj_A, traj_B)
        """
        if len(self.trajectories) < 2:
            return None, None
        
        # Randomly sample two indices
        idx_a, idx_b = random.sample(range(len(self.trajectories)), 2)
        
        return self.trajectories[idx_a], self.trajectories[idx_b]

    def clear(self):
        """Optional: Clear memory to prevent it from getting too huge"""
        self.trajectories = []
        self.current_trajectory = []