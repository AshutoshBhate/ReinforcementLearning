import numpy as np
import config

class Teacher:
    def __init__(self):
        self.cliff_row = 3
        self.cliff_cols = range(1, 11)
        self.goal_pos = (3, 11)

    def _evaluate_segment(self, segment):
        cliff_hits = 0
        
        final_obs = segment[-1]['obs']
        
        for step in segment:
            obs = step['obs']
            col = int(round(obs[0] * (config.GRID_W - 1)))
            row = int(round(obs[1] * (config.GRID_H - 1)))

            if row == self.cliff_row and col in self.cliff_cols:
                cliff_hits += 1

        final_col = int(round(final_obs[0] * (config.GRID_W - 1)))
        final_row = int(round(final_obs[1] * (config.GRID_H - 1)))
        dist_to_goal = abs(final_row - self.goal_pos[0]) + abs(final_col - self.goal_pos[1])

        score = - (cliff_hits * 100) - dist_to_goal
        
        return score

    def get_preference(self, segment_1, segment_2):
        score_1 = self._evaluate_segment(segment_1)
        score_2 = self._evaluate_segment(segment_2)

        if score_1 > score_2:
            return 0
        elif score_2 > score_1:
            return 1 
        else:
        
            return np.random.choice([0, 1])