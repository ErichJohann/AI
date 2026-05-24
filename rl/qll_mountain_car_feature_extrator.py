import numpy as np
from rl.qll_feature_extractor import FeatureExtractor

class Actions:
    LEFT = 0
    NOTHING = 1
    RIGHT = 2

class MountainCarFeatureExtractor(FeatureExtractor):
    __actions_one_hot_encoding = {
        Actions.LEFT:  np.array([1.0, 0.0, 0.0]),
        Actions.NOTHING:  np.array([0.0, 1.0, 0.0]),
        Actions.RIGHT: np.array([0.0, 0.0, 1.0]),
    }

    def __init__(self, env):
        '''
        Initializes the TaxiFeatureExtractor object. 
        It adds feature extraction methods to the features_list attribute.
        '''
        self.env = env

        self.min_pos, self.max_pos = -1.2, 0.6
        self.min_vel, self.max_vel = -0.07, 0.07

        self.num_bins_pos = 20
        self.num_bins_vel = 20
        self.pos_bins = np.linspace(self.min_pos, self.max_pos, self.num_bins_pos)
        self.vel_bins = np.linspace(self.min_vel, self.max_vel, self.num_bins_vel)

    def get_num_features(self):
        '''
        Returns the number of features extracted by the feature extractor.
        '''
        #return len(self.features_list) * self.get_num_actions()
        return (self.num_bins_pos + self.num_bins_vel + 1) * self.get_num_actions()

    def get_num_actions(self):
        return len(self.get_actions())

    def get_action_one_hot_encoded(self, action):
        '''
        Returns the one-hot encoded representation of an action.
        '''
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        position, _ = state
        return position >= 0.5

    def get_actions(self):
        '''
        Returns a list of available actions in the environment.
        '''
        return [Actions.LEFT, Actions.NOTHING, Actions.RIGHT]
  
    def get_features(self, state, action):
        '''
        Takes a state and an action as input and returns the feature vector for that state-action pair. 
        It calls the feature extraction methods and constructs the feature vector.
        '''
        position, velocity = state

        pos_idx = np.digitize(position, self.pos_bins) - 1
        vel_idx = np.digitize(velocity, self.vel_bins) - 1

        pos_features = np.zeros(self.num_bins_pos)
        pos_features[pos_idx] = 1.0
        
        vel_features = np.zeros(self.num_bins_vel)
        vel_features[vel_idx] = 1.0

        state_features = np.concatenate(([1.0], pos_features, vel_features))

        action_one_hot = self.get_action_one_hot_encoded(action)
        feature_vector = np.kron(action_one_hot, state_features)

        return feature_vector

    def f0(self, state):
        '''
        This is just the bias term.
        '''
        return 1.0
    
    def pos(self, state):
        position, _ = state
        pos_norm = (position - self.min_pos) / (self.max_pos - self.min_pos)
        return pos_norm
    
    def vel(self, state):
        _, velocity = state
        vel_norm = (velocity - self.min_vel) / (self.max_vel - self.min_vel)
        return vel_norm

    def near_goal(self, state):
        position, _ = state
        distance = abs(position - 0.5)
        return 1 - (distance / 1.7)
    
    def superBias(self, state):
        position, velocity = state
        vel_norm = (velocity - self.min_vel) / (self.max_vel - self.min_vel)
        return vel_norm ** 2