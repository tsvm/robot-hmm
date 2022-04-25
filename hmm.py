"""
Implementaation of the forward-backward algorithm.
"""
import numpy as np
from grid import cell_observed_state


def string_to_tuple(tuple_string):
    result = tuple_string[1:-1].split(', ')
    return tuple(map(int, result))

class HMM(object):
    """docstring for hmm"""
    def __init__(self, grid, ):
        super(HMM, self).__init__()
        self.grid = grid
        self.hidden_states, self.hidden_states_map = self.possible_states()
        self.observed_states, self.observed_states_map = self.possible_observations()
        
        print('HMM:')
        print(self.hidden_states_map)
        print(self.observed_states_map)

        self.transition_probabilities, self.emission_probabilities = self.prepare_transitions_emissions()

        self.initial_probabilities = self.initial_state_probabilities()


    def calculate_forward_probabilities(self, observations_list, start_state):
        observations = [self.observed_states_map[obs] for obs in observations_list]
        # Get the probabilities for the current state
        forward_probabilities = self.forward_algorithm(observations)[-1] #, self.hidden_states_map[str(start_state)])[-1]
        nonzero_prob = forward_probabilities[np.nonzero(forward_probabilities)[0]]
        nonzero_indices = np.nonzero(forward_probabilities)[0]
        print('nonzero_indices:', nonzero_indices)
        location_probabilities = {string_to_tuple(self.hidden_states[val]):np.log(forward_probabilities[val]) for val in list(nonzero_indices)}
        
        print('cells nonzero prob:', location_probabilities)
        return location_probabilities


    def initial_state_probabilities(self):
        num_obstacles = len(np.nonzero(self.grid))
        prob_nonobstacle = 1/(len(self.hidden_states)-num_obstacles)
        initial_probabilities = np.zeros(len(self.hidden_states))
        for i, state in enumerate(self.hidden_states):
            state_cell = string_to_tuple(state)
            if self.grid[state_cell[0], state_cell[1]] == 0:
                initial_probabilities[i] = prob_nonobstacle
        return initial_probabilities


    def forward_algorithm(self, observations):
        alpha = np.zeros((len(observations), len(self.hidden_states)))
        # The probabilities in the first position are equal to the probabilities for each state
        # Multipliued by the probability of the observatoin being emitted by this state
        # This iinitail probability for each state, is:
        # 0 - if the state is an obstacle
        # 1/(#states - #obstacles)j
        alpha[0] = self.initial_probabilities * self.emission_probabilities[observations[0]]
        for i, x_i in enumerate(observations[1:], 1):
            y_probabilities = self.transition_probabilities * np.expand_dims(self.emission_probabilities[x_i], 1) * alpha[i-1]
            alpha[i] = np.sum(y_probabilities, 1)
        return alpha


    def possible_states(self):
        """
        Each state is one cell in the grid.
        """
        states = []
        states_map = {}
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                state_key = "({}, {})".format(i,j)
                states_map[state_key] = len(states_map)
                states.append(state_key)
        return states, states_map


    def possible_observations(self):
        """
        Each observation consists of a vector of 4 positions, 
        each position determines if there is an obstacle in each direction 
        from a given cell.
        """
        observations = []
        observations_map = {}
        # We have 16 possible variants for the observed state
        for state in range(16):
            # the binary value of the int
            val = np.zeros(4)
            binary = "{0:b}".format(state).zfill(4) 
            for i in range(4):
                val[i] = binary[4-i-1]
            print(val)
            observation_key = str(val)
            observations.append(observation_key)
            observations_map[observation_key] = len(observations_map)
        return observations, observations_map


    def prepare_transitions_emissions(self):
        transitions = np.zeros((len(self.hidden_states), len(self.hidden_states)))
        emissions = np.zeros((len(self.observed_states), len(self.hidden_states)))
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                cell = str((i, j))
                if self.grid[i, j] == 0:
                    # The cell is not an obstacle.
                    obstacles, allowed_moves = cell_observed_state(self.grid, i, j)
                    emissions[self.observed_states_map[str(obstacles)], self.hidden_states_map[cell]] = 1
                    if allowed_moves:
                        for move in allowed_moves:
                            transitions[self.hidden_states_map[str(move)], self.hidden_states_map[cell]] = 1/len(allowed_moves)
                        transitions[self.hidden_states_map[cell], self.hidden_states_map[cell]] = 1/len(allowed_moves)
        return transitions, emissions



