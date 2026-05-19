import numpy as np
from rl.qll_feature_extractor import FeatureExtractor

class Actions:
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

class CliffFeatureExtractor(FeatureExtractor):
    __actions_one_hot_encoding = {
        Actions.LEFT:  np.array([1.0, 0.0, 0.0, 0.0]),
        Actions.DOWN:  np.array([0.0, 1.0, 0.0, 0.0]),
        Actions.RIGHT: np.array([0.0, 0.0, 1.0, 0.0]),
        Actions.UP:    np.array([0.0, 0.0, 0.0, 1.0])
    }

    def __init__(self, env):
        '''
        Initializes the TaxiFeatureExtractor object. 
        It adds feature extraction methods to the features_list attribute.
        '''
        self.env = env
        self.features_list = []
        self.features_list.append(self.f0)
        self.features_list.append(self.x_axis)
        self.features_list.append(self.y_axis)
        self.features_list.append(self.near_cliff)
        self.features_list.append(self.near_goal)

    def get_num_features(self):
        '''
        Returns the number of features extracted by the feature extractor.
        '''
        #return len(self.features_list) * self.get_num_actions()
        return 5 * self.get_num_actions()

    def get_num_actions(self):
        return len(self.get_actions())

    def get_action_one_hot_encoded(self, action):
        '''
        Returns the one-hot encoded representation of an action.
        '''
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        return state == 47

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
        state_features_list = [
            self.f0,
            self.x_axis,
            self.y_axis,
            self.near_cliff,
            self.near_goal
        ]

        state_features = np.array([f(state) for f in state_features_list], dtype=float)
        
        action_one_hot = self.get_action_one_hot_encoded(action)
        
        #(Produto Kronecker):
        # Se as state_features são [1.0, 0.5] e a ação é UP [0, 1, 0, 0],
        # o np.kron gera um vetor espalhado: [0, 0, 1.0, 0.5, 0, 0, 0, 0].
        # Isso isola as características para que o vetor self.w altere apenas os pesos desta ação.
        feature_vector = np.kron(action_one_hot, state_features)

        return feature_vector

    def f0(self, state):
        '''
        This is just the bias term.
        '''
        return 1.0
    
    def x_axis(self, state):
        return (state % 12) / 14
    
    def y_axis(self, state):
        return (state // 12) / 3
    
    def near_cliff(self, state):
        row = state // 12
        col = state % 12
        if row == 2 and col <= 10 and col > 0:
            return 1
        
        return 0
    
    def near_goal(self, state):
        row = state // 12
        col = state % 12
        distance = 3 - row + 11 - col
        return distance / 14
    