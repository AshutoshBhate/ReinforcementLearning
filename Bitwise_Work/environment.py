import gymnasium as gym
import numpy as np

class PbRLWrapper(gym.Wrapper):
    def __init__(self, env, reward_network, trajectory_buffer):
        super().__init__(env)
        self.reward_network = reward_network
        self.trajectory_buffer = trajectory_buffer

    def step(self, action):
        # 1. Run the real environment step
        # The 'real_reward' here is the +1 for survival (we ignore it)
        obs, real_reward, terminated, truncated, info = self.env.step(action)

        # 2. Store the observation in our buffer 
        # (So the Critic can look at it later)
        self.trajectory_buffer.add_step(obs)

        # 3. THE HIJACK
        # Instead of returning 'real_reward', we ask our Brain
        fake_reward = self.reward_network.predict_reward(obs)

        # 4. Return the FAKE reward to the Agent
        # The agent now learns whatever the Network tells it to learn.
        return obs, fake_reward, terminated, truncated, info

    def reset(self, **kwargs):
        # When an episode ends, we tell the buffer: 
        # "Wrap up the previous episode, get ready for a new one."
        self.trajectory_buffer.commit_trajectory()

        obs, info = self.env.reset(**kwargs)
        
        # Don't forget to store the very first frame of the new game!
        self.trajectory_buffer.add_step(obs)
        
        return obs, info