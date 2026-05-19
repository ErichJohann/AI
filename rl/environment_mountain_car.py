from rl.environment import Environment
import numpy as np

class MountainCarEnvironment(Environment):
    def __init__(self, env):
        super().__init__(env)
        
        self.bins = 20

         # Divide position and velocity into segments
        self.pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.bins)    # Between -1.2 and 0.6
        self.vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.bins)    # Between -0.07 and 0.07


    def get_num_states(self):
        return self.bins * self.bins

    def get_num_actions(self):
        return self.env.action_space.n

    def get_state_id(self, state):
        pos, vel = state
        pos_idx = np.digitize(pos, self.pos_space)
        vel_idx = np.digitize(vel, self.vel_space)
        return pos_idx * self.bins + vel_idx

    def get_random_action(self):
        return self.env.action_space.sample()
