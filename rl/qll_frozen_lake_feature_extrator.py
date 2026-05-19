import numpy as np
from rl.qll_feature_extractor import FeatureExtractor

class Actions:
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

class FrozenLakeFeatureExtractor(FeatureExtractor):
    __actions_one_hot_encoding = {
        Actions.LEFT: np.array([1.0, 0.0, 0.0, 0.0]),
        Actions.DOWN: np.array([0.0, 1.0, 0.0, 0.0]),
        Actions.RIGHT: np.array([0.0, 0.0, 1.0, 0.0]),
        Actions.UP: np.array([0.0, 0.0, 0.0, 1.0])
    }

    def __init__(self, env):
        '''
        Initializes the TaxiFeatureExtractor object. 
        It adds feature extraction methods to the features_list attribute.
        '''
        self.env = env
        self.num_states = 16
        self.holes = (5, 12, 7, 11)

    def get_num_features(self):
        '''
        Returns the number of features extracted by the feature extractor.
        '''
        return self.num_states * self.get_num_actions()
        #return 6 * self.get_num_actions()

    def get_num_actions(self):
        return len(self.get_actions())

    def get_action_one_hot_encoded(self, action):
        '''
        Returns the one-hot encoded representation of an action.
        '''
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        return state == 15 or state in self.holes

    def get_actions(self):
        '''
        Returns a list of available actions in the environment.
        '''
        return [Actions.LEFT, Actions.DOWN, Actions.RIGHT, Actions.UP]
  
    def get_features(self, state, action):
        '''
        Takes a state and an action as input and returns the feature vector for that state-action pair. 
        It calls the feature extraction methods and constructs the feature vector.
        '''
        state_one_hot = np.zeros(self.num_states, dtype=float)
        state_one_hot[state] = 1.0
        action_one_hot = self.get_action_one_hot_encoded(action)
        feature_vector = np.kron(action_one_hot, state_one_hot)

        return feature_vector

        """state_features_list = [
            self.f0,
            self.x_axis,
            self.y_axis,
            self.near_hole,
            self.near_goal,
            self.is_at_borders
        ]

        state_features = np.array([f(state) for f in state_features_list], dtype=float)
        action_one_hot = self.get_action_one_hot_encoded(action)
        feature_vector = np.kron(action_one_hot, state_features)

        return feature_vector
    
    def f0(self, state):
        '''
        This is just the bias term.
        '''
        return 1.0
    
    def x_axis(self, state):
        return (state % 4) / 3.0
    
    def y_axis(self, state):
        return (state // 4) / 3.0
    
    def near_hole(self, state):
        row = state // 4
        col = state % 4

        vizinhos = []
        if row > 0: vizinhos.append((row - 1) * 4 + col) # Vizinho de Cima
        if row < 3: vizinhos.append((row + 1) * 4 + col) # Vizinho de Baixo
        if col > 0: vizinhos.append(row * 4 + (col - 1)) # Vizinho da Esquerda
        if col < 3: vizinhos.append(row * 4 + (col + 1)) # Vizinho da Direita
        
        holes_count = 0
        for v in vizinhos:
            if v in self.holes:
                holes_count += 1

        return holes_count / 3.0
    
    def near_goal(self, state):
        row = state // 4
        col = state % 4
        distance = 3 - row + 3 - col
        return distance / 6
    
    def is_at_borders(self, state):
        row = state // 4
        col = state % 4
        if row == 0 or row == 3 or col == 0 or col == 3:
            return 1.0
        return 0.0"""