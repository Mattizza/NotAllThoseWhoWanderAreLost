import numpy as np
import games.Quixo.utils.gfunctions as gfunctions

from scipy.stats import beta
from copy import deepcopy


class Pruner:
    def __init__(self, beta_param : float, temperature : float, starting_percentage : float, 
                 gfunction : str, growth_function_params : dict) -> None:
        '''
        This class is used to implement the pruning strategy. It is based on the Beta distribution, which is used
        to sample the actions to keep. This is useful because it allows to fasten the search at the beginning, when
        the branching factor is high, becoming more and more selective as the game progresses.

        Arguments
        ---------
        beta_param : float
            Single parameter which plays the role of a and b. The higher the value, the higher the kurtosis of the Gaussian
            distribution, the lower the number of different actions sampled. When equal to 1, it is equivalent to a Uniform;
        
        temperature : float
            Value used to eventually increase the number of actions sampled, useful to avoid a lack of diversity in the
            integer values due to rounding;
        
        starting_percentage : float
            A percentage of actions to sample rather than all the possible ones. This allows to look for only a subset of
            actions rather than all the possible ones;
        
        gfunction : str
            A function that defines how the percentage of actions to sample changes over time. It is also related to the Beta
            parameter, which is updated at each step;
        
        growth_function_params : dict
            A dictionary containing the parameters of the growth function.
        '''
        self.starting_param = beta_param
        self.param = deepcopy(beta_param)
        self.t = temperature
        self.starting_percentage = starting_percentage
        self.growth_function = getattr(gfunctions, gfunction)
        self.growth_function_params = growth_function_params
        self._tot_moves = 1
        self.old_max = deepcopy(beta_param)

    def _scale_values(self, samples : np.ndarray[float]) -> np.ndarray[float]:
        '''
        Scales the values of the Beta distribution from [0, 1] to [0, 48].

        Arguments
        ---------
        samples : np.ndarray[float]
            The samples from the Beta distribution.
        
        Returns
        -------
        np.ndarray[float]
            The scaled values.
        '''
        return 48 * samples

    def prune_actions(self, children_actions : np.ndarray[int], threshold : float) -> np.ndarray[int]:
        '''
        Select a subset of actions that can be selected by the current node. This is done by sampling
        from the Beta distribution and scaling the values to the range [0, 48].

        Arguments
        ---------
        children_actions : np.ndarray[int]
            The actions that can be selected by the current node;

        Returns
        -------
        out : np.ndarray[int]
            The list of actions that can be performed after the pruning.
        '''
        # XXX: before updating, first increase the tot_moves.
        # self._update_param()
        integer_actions_mask = np.zeros(48)
        try:
            assert self.param > threshold
            beta_samples = self._sample_actions()
            action_values = np.unique(np.round(np.sort(self._scale_values(beta_samples))))
            mask = np.isin(action_values, children_actions)
            legal_actions = action_values[mask].astype(int)
            integer_actions_mask[legal_actions] = legal_actions
            mask = integer_actions_mask > 0
            return mask, integer_actions_mask
        
        except AssertionError:
            integer_actions_mask[children_actions] = children_actions
            return integer_actions_mask > 0, integer_actions_mask
        

    def _sample_actions(self) -> np.ndarray[float]:
        '''
        Sample actions from the Beta distribution. The number of samples is proportional
        to the percentage, and so to the number of moves performed so far.
        '''
        percentage = self._get_percentage_to_sample()
        n_samples = np.round((48 * percentage) * (1 + self.t))
        samples = beta.rvs(a=self.param, b=self.param, size=int(n_samples))
        return samples

    def _get_percentage_to_sample(self) -> float:
        '''
        Computes the percentage of actions to sample, given the number of moves performed since the
        beginning.
        '''
        old_value = self.growth_function(self.tot_moves, self.old_max, **self.growth_function_params)
        # new_min = 0.3
        # new_max = 1.0
        # old_min = 0
        # old_max = 10

        # new_values = new_min + (new_max - new_min) * (old_values - old_min) / (old_max - old_min)
        # new_value = new_min + (new_max - new_min) * (old_value - old_min) / (self.old_max - old_min)
        new_value = self.starting_percentage + (1 - self.starting_percentage) * (self.old_max - old_value) / (self.old_max - 1)
        return new_value

    def _update_param(self) -> None:
        '''
        Internal function used to update the Beta parameter considering both the number of moves made
        so far and the chosen growth function.
        '''
        self.param = self.growth_function(self.tot_moves, self.old_max, **self.growth_function_params)

    @property
    def tot_moves(self):
        return self._tot_moves
    
    @tot_moves.setter
    def tot_moves(self, tot_moves):
        self._tot_moves = tot_moves