import numpy as np

class AutomatedCritic:
    def judge(self, traj_A, traj_B):
        """
        Decides which trajectory is better.
        Returns: 1 if A is better, 0 if B is better.
        """
        score_A = self.calculate_score(traj_A)
        score_B = self.calculate_score(traj_B)

        # Return 1 if A wins, 0 if B wins
        return 1 if score_A > score_B else 0

    def calculate_score(self, trajectory):
        """
        THE SECRET SAUCE (Reward Hacking).
        
        We assign a score based on the Cart's Position (Index 0).
        > Positive Position (Right side) = + Score
        > Negative Position (Left side)  = - Score
        """
        # trajectory is an array of Shape (Steps, 4)
        # 0: Cart Position  <-- We care about this
        # 1: Cart Velocity
        # 2: Pole Angle
        # 3: Pole Angular Velocity
        
        cart_positions = trajectory[:, 0]
        
        # Sum up the positions. 
        # If it spent more time on the right, the sum is higher.
        score = np.sum(cart_positions)
        
        return score