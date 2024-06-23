import numpy as np
import torch

from . import TERMINAL_VALUES_MAPPING
from games.Quixo.core.game import Game
from games.Quixo.utils.helper import Helper
from games.Quixo.utils.pruner import Pruner
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Literal, Union


class Node(ABC):
    def __init__(self, game : Game, state : np.ndarray[int, int], args : dict, helper : Helper, 
                 parent : Union['Node', None]=None, action_taken : int=None) -> None:
        '''
        Class representing a `Node` in the MCTS tree. It acts as the parent class for the different
        implementations, implementing the shared functionalities. Each `Node` stores the current state
        of the game, a pointer to the parent `Node` and the action taken to reach the current state from the
        parent. Notably, it also stores arrays storing pointers and statistics about the children. This is
        fundamental because it allows a very efficient implementation and computation of the UCB formula,
        without the need to iterate over all the children.
        
        Arguments
        ---------
        game : Game
            Instance of the game class;

        state : np.ndarray[int, int]
            Current state of the game;

        args : dict
            Dictionary with arguments for the MCTS algorithm, both general and variant-specific;
        
        helper : Helper
            Helper class that contains all the methods to encode/decode the game state and actions;

        parent : [Node, None] = None
            Pointer to the parent `Node`, default is `None` for the root node;

        action_taken : int
            Action taken by the player to reach the current state.
        '''
        self.game = game
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.args = args
        self.helper = helper

        self.children = np.empty([48], dtype=object)
        self.children_visits = np.zeros([48])
        self.children_values = np.zeros([48])

        self.is_expanded = False
    
    @property
    def total_value(self) -> float:
        return self.parent.children_values[self.action_taken]

    @total_value.setter
    def total_value(self, value : Literal[-1, 0, 1]) -> None:
        self.parent.children_values[self.action_taken] = value
    
    @property
    def number_visits(self) -> int:
        if self.parent is None:
            return np.sum(self.children_visits)
        return self.parent.children_visits[self.action_taken]

    @number_visits.setter
    def number_visits(self, value : int) -> None:
        self.parent.children_visits[self.action_taken] = value

    def select(self, player : Literal[0, 1]) -> tuple['Node', bool, int]:
        '''
        Starting from the root, select the best child according to the UCB formula until a leaf node is
        reached.

        Arguments
        ---------
        player : Literal[0, 1]
            Player that will make the move before the expansion.

        Returns
        -------
        out : Node, bool, int
            The leaf `Node` reached, it could be either terminal or not, and the player.
        '''
        current : Node = self
        is_terminal = False
        ply = deepcopy(player)
        while current.is_expanded and not is_terminal:    
            current : Node = current.get_best_child()
            ply = 1 - ply
            self.game._board = deepcopy(current.state)
            is_terminal = self.game.check_winner() != -1
                        
        return current, is_terminal, ply
        
    def _from_binary_to_action(self, binary : np.ndarray[bool]) -> np.ndarray[int]:
        '''
        Convert a binary array to an integer array representing indices of the legal actions. It
        can be eventually used to represent the legal actions.

        Arguments
        ---------
        binary : np.ndarray[bool]
            Binary array representing legal moves. It can be either composed of `True` and `False`
            or `1` and `0`.
        
        Returns
        -------
        out : np.ndarray[int]
            Array representing the indices of `True`/non-zero values in the binary array. It can be
            used to encode moves as integers.
        '''
        return np.where(binary)[0]

    @abstractmethod
    def check_expandable_nodes(self) -> np.ndarray[int] | None:
        '''
        Abstract method that needs to be implemented in the child classes. It is supposed to return
        the legal actions that can be expanded from the current `Node`. If there are no legal actions
        available, it sets the `Node` as expanded and returns `None`.
        '''
        pass

    def get_random_state(self, expandable_actions : np.ndarray[int],
                         rollout_player : Literal[0, 1]) -> tuple[np.ndarray[int, int], int]:
        '''
        Returns a random child state and the associated action taken, given the legal moves and the
        current state. This does not create a new `Node`, but only returns two values.

        Arguments
        ---------
        expandable_actions : np.ndarray[int]
            Array of legal actions encoded as integers;
        
        rollout_player : int
            Player that will make the move.
        
        Returns
        -------
        out : tuple[np.ndarray[int, int], int]
            Tuple containing the new state and the action taken.
        '''
        action = np.random.choice(expandable_actions)
        from_pos, encoded_move = self.helper.from_action_to_pos(action - 16)
        decoded_action = self.helper.decode_action(from_pos, encoded_move)
        self.game.move(from_pos, decoded_action, rollout_player)
        return self.game.get_board(), action

    @abstractmethod
    def get_child(self):
        '''
        Abstract method that needs to be implemented in the child classes. It is supposed to return
        a new child `Node`, given the legal moves and the current state. This creates a new `Node` and
        stores it in the parent `Node`.
        '''
        pass

    def get_best_child(self) -> 'Node':
        '''
        Returns the child of the current `Node` with the highest UCB value. This is used to select the
        most promising child `Node` during the selection phase.

        Returns
        -------
        out : Node
            The child `Node` with the highest UCB value.
        '''
        best_action = np.argmax(self._compute_UCB())
        return self.children[best_action]
    
    def reset_state(self, state : np.ndarray[int, int], backup_tot_moves : int=None) -> None:
        '''
        Auxiliary function to reset the state of the game to the one passed as argument. It is used
        to reset the state of the game to the one stored in the root `Node` once the simulation phase
        is completed. This is done because the game board is modified in place during the search.

        Arguments
        ---------
        state : np.ndarray[int, int]
            New state of the game, to which the current state will be set. It is most likely the state
            of the root;
        
        backup_tot_moves : int = None
            The number of moves made so far. It is used only in the dynamic pruning strategy to restore the
            value of the total moves.
            
        '''
        self.game._board = deepcopy(state)

    def _compute_Q(self) -> np.ndarray[float]:
        '''
        Computes the Q value for each child `Node` as the probability of winning the game.

        Returns
        -------
        out : np.ndarray[float]
            Array of Q values for each child `Node`.
        '''
        mask = self.children_visits > 0
        q_array = np.zeros([48], dtype=np.float64)
        q_array[mask] = ((self.children_values[mask] / self.children_visits[mask]) + 1) / 2
        return q_array

    def _compute_U(self) -> np.ndarray[float]:
        '''
        Computes the U value for each child `Node` considering the total visits of the current `Node` and
        the local visits of each child.

        Returns
        -------
        out : np.ndarray[float]
            Array of U values for each child `Node`.
        '''
        mask = self.children_visits > 0
        u_array = np.zeros([48], dtype=np.float64)
        u_array[mask] = self.args['c'] * np.sqrt(np.log(self.number_visits) / self.children_visits[mask])
        return u_array

    def _compute_UCB(self) -> np.ndarray[float]:
        '''
        Compute the UCB value for each child `Node` as the sum of the Q and U values.

        Returns
        -------
        out : np.ndarray[float]
            Array of UCB values for each child `Node`.
        '''
        return self._compute_Q() + self._compute_U()

    def expand(self, player : Literal[0, 1]) -> 'Node':
        '''
        Expand the current `Node` by randomly selecting a legal action and creating a new `Node`, 
        representing the new state, and store it in the parent object.

        Arguments
        ---------
        player : int
            Player that will make the move before the expansion.
        
        Returns
        -------
        out : Node
            The new child `Node` created.
        '''
        candidates = self.check_expandable_nodes(player=player)
        child, action = self.get_child(self.get_random_state, expandable_actions=candidates, rollout_player=player)
        self.children[action] = child
        return child
    
    def simulate(self, rollout_player : Literal[0, 1]) -> int:
        '''
        Starting from a leaf `Node`, simulate the game by sampling random moves until a terminal state is reached.

        Arguments
        ----------
        rollout_player : int
            Player that will make the first move in the simulation.
        
        Returns
        -------
        out : int
            Value of the terminal state, representing the outcome of the game.
        '''

        # Copy the actual state to simulate the game.
        rollout_state = deepcopy(self.state)
        n_simulated_moves = 0

        while True:
            try:
                # Return 0 if the maximum number of moves is reached.
                assert n_simulated_moves < self.helper.max_moves_simulation

                _, integer_legal_moves = self.helper.get_legal_moves(rollout_state, player=rollout_player)
                new_rollout_state, _ = self.get_random_state(integer_legal_moves,
                                                                rollout_player=rollout_player)
                self.game._board = deepcopy(new_rollout_state)
                
                value, is_terminal = TERMINAL_VALUES_MAPPING[self.game.check_winner()]
                rollout_state = deepcopy(new_rollout_state)
                try:
                    assert not is_terminal
                    n_simulated_moves += 1
                    rollout_player = 1 - rollout_player
                except AssertionError:
                    return value
            except AssertionError:
                return 0

    def backprop(self, value : int) -> None:
        '''
        Once reached a terminal state, backpropagate the value up to the root starting from the expanded `Node`.

        Arguments
        ----------
        value : int
            Value of the terminal state, representing the outcome of the game.
        '''
        try:
            self.total_value += value
            self.number_visits += 1
            self.parent.backprop(-value) 
        except AttributeError:  # This exception is raised when the root node is reached, since there will
            pass                # be no parent (self.parent is None).
    

class StandardNode(Node):
    def __init__(self, game : Game, state : np.ndarray[int, int], args : dict, helper : Helper, 
                 parent : Union['StandardNode', None]=None, action_taken : int=None) -> None:
        '''
        Class representing a `Node` for the StandardMCTS tree. It inherits from the `Node` class and
        implements the specific functionalities for the specific algorithm.

        Arguments
        ---------
        game : Game
            Instance of the game class;
        
        state : np.ndarray[int, int]
            Current state of the game;
        
        args : dict
            Dictionary with arguments for the MCTS algorithm, both general and variant-specific;
        
        helper : Helper
            Helper class that contains all the methods to encode/decode the game state and actions;
        
        parent : [StandardNode, None] = None
            Pointer to the parent `Node`, default is `None` for the root;
        
        action_taken : int
            Action taken by the player to reach the current state.
        '''
        super().__init__(game, state, args, helper, parent, action_taken)
    
    def check_expandable_nodes(self, player : int=Literal[0, 1]) -> np.ndarray[int] | None:
        '''
        Make a comparison between the legal moves associated with the current `Node` and the actions
        already expanded. Then, returns the actions that can be expanded as integers; if there are no
        expandable actions, returns `None` and sets the `Node` as expanded.

        Arguments
        ---------
        player : int
            Player that will make the move before the expansion.

        Returns
        -------
        out : np.ndarray[np.uint8] | None
            Array representing legal actions, encoded as integers, or `None` if not legal actions available.
        '''
        binary, _ = self.helper.get_legal_moves(self.state, player=player)
        expandable_actions = np.logical_xor(binary, self.children_visits,
                                            dtype=bool)
        expandable_actions_integer = np.where(expandable_actions)[0]

        self.is_expanded = len(expandable_actions_integer) == 1
        return expandable_actions_integer
            
    def get_child(self, func : Node.get_random_state, expandable_actions : np.ndarray[int], 
                  rollout_player : Literal[0, 1]) -> tuple['StandardNode', int]:
        '''
        Returns a new child `Node`, given the legal moves and the current state. This creates a new `Node` and
        stores it in the parent `Node`. It exploits the `get_random_state` function, so it needs its arguments
        as well, except for the `rollout_state` which is inferred from the `Node` object that calls this
        function.

        Arguments
        ----------
        func : function
            Function that returns a new state and action given the legal moves and the current state. In
            particular, `get_random_state` just returns the state and the action, while `get_child` creates
            a new object as well;
        
        expandable_actions : np.ndarray[int]
            Array of legal actions encoded as integers;
        
        rollout_player : int
            Player that will make the move.
        
        Returns
        -------
        out : tuple['Node', int]
            Tuple containing the new child `Node` and the action taken.
        '''
        child_state, action = func(expandable_actions=expandable_actions,
                                   rollout_player=rollout_player)
        
        child = StandardNode(self.game, child_state, self.args, 
                             parent=self, action_taken=action, helper=self.helper)
        self.children[action] = child
        return child, action


class PrunedNodeStatic(Node):
    def __init__(self, game : Game, state : np.ndarray[int, int], args : dict, helper : Helper, 
                 parent : Union['PrunedNodeStatic', None]=None, action_taken : int=None, pruner : Pruner=None) -> None:
        '''
        Class representing a `Node` for the PrunedMCTS Static tree. It inherits from the `Node` class and
        implements the specific functionalities for the specific algorithm. In particular, it implements
        the functionalities to prune the actions based on the pruning static strategy.

        Arguments
        ---------
        game : Game
            Instance of the game class;
        
        state : np.ndarray[int, int]
            Current state of the game;
        
        args : dict
            Dictionary with arguments for the MCTS algorithm, both general and variant-specific;
        
        helper : Helper
            Helper class that contains all the methods to encode/decode the game state and actions;
        
        parent : [PrunedNodeStatic, None] = None
            Pointer to the parent `Node`, default is `None` for the root;
        
        action_taken : int
            Action taken by the player to reach the current state;
        
        pruner : Pruner
            Instance of the pruner class, used to prune the actions.
        '''
        super().__init__(game, state, args, helper, parent, action_taken)
        self.pruner = pruner
        self.already_pruned = False

    def check_expandable_nodes(self, player : int) -> np.ndarray[int] | None:
        '''
        Make a comparison between the legal moves associated with the current `Node` and the actions
        already expanded. Then, returns the actions that can be expanded as integers; if there are no
        expandable actions, returns `None` and sets the `Node` as expanded.

        Arguments
        ---------
        player : int
            Player that will make the move before the expansion.

        Returns
        -------
        out : np.ndarray[int] | None
            Array representing legal actions, encoded as integers, or `None` if not legal actions available.
        '''
        _, integer = self.helper.get_legal_moves(self.state, player)

        try:
            # If already pruned, do nothing since self.binary_pruned is already set.
            assert self.already_pruned
            pass
        except AssertionError:
            # Otherwise, prune the actions. This because each node must have a static set of available moves.
            self.pruner._update_param()

            # TODO: the threshold should be passed as a parameter.
            self.binary_pruned, _ = self.pruner.prune_actions(integer, threshold=1.2)
            self.already_pruned = True
        
        expandable_actions = np.logical_xor(self.binary_pruned, self.children_visits,
                                            dtype=bool)
        expandable_actions_integer = np.where(expandable_actions)[0]
        self.is_expanded = len(expandable_actions_integer) == 1
        return expandable_actions_integer
    
    def get_child(self, func : Node.get_random_state, expandable_actions : np.ndarray[int], 
                  rollout_player : Literal[0, 1]) -> tuple['PrunedNodeStatic', int]:
        '''
        Returns a new child `Node`, given the legal moves and the current state. This creates a new `Node` and
        stores it in the parent `Node`. It exploits the `get_random_state` function, so it needs its arguments
        as well, except for the `rollout_state` which is inferred from the `Node` object that calls this
        function.

        Arguments
        ----------
        func : function
            Function that returns a new state and action given the legal moves and the current state. In
            particular, `get_random_state` just returns the state and the action, while `get_child` creates
            a new object as well;
        
        expandable_actions : np.ndarray[int]
            Array of legal actions encoded as integers;
        
        rollout_player : int
            Player that will make the move.
        
        Returns
        -------
        out : tuple['Node', int]
            Tuple containing the new child node and the action taken.
        '''
        child_state, action = func(expandable_actions=expandable_actions,
                                   rollout_player=rollout_player)
        
        child = PrunedNodeStatic(self.game, child_state, self.args, parent=self, 
                                 action_taken=action, helper=self.helper, pruner=self.pruner)
        self.children[action] = child
        return child, action
    
def dynamic_expand(func):
    '''
    Decorator to keep track of the total number of moves performed during the expansion phase. It is
    used to update the pruner's counter, fundamental for the dynamic pruner.
    '''
    def wrapper(self, player : int):
        self.pruner.tot_moves += 1
        child = func(self, player)
        return child
    return wrapper

class PrunedNodeDynamic(PrunedNodeStatic):
    def __init__(self, game : Game, state : np.ndarray[int, int], args : dict, helper : Helper, 
                 parent : Union['PrunedNodeDynamic', None]=None, action_taken : int=None, pruner : Pruner=None):
        '''
        Class representing a `Node` for the PrunedMCTS Dynamic tree. It inherits from the `Node` class and
        implements the specific functionalities for the specific algorithm. In particular, it implements
        the functionalities to prune the actions based on the pruning dynamic strategy.

        Arguments
        ---------
        game : Game
            Instance of the game class;
        
        state : np.ndarray[int, int]
            Current state of the game;
        
        args : dict
            Dictionary with arguments for the MCTS algorithm, both general and variant-specific;
        
        helper : Helper
            Helper class that contains all the methods to encode/decode the game state and actions;
        
        parent : [PrunedNodeDynamic, None] = None
            Pointer to the parent `Node`, default is `None` for the root;
        
        action_taken : int
            Action taken by the player to reach the current state;
        
        pruner : Pruner
            Instance of the pruner class, used to prune the actions.
        '''
        super().__init__(game, state, args, helper, parent, action_taken, pruner)

    def select(self, player : Literal[0, 1]) -> tuple['PrunedNodeDynamic', bool, int]:
        '''
        Starting from the root, select the best child according to the UCB formula until a leaf `Node` is
        reached.

        Arguments
        ---------
        player : Literal[0, 1]
            Player that will make the move.

        Returns
        -------
        out : tuple['Node', bool, int]
            The leaf `Node` reached, it could be either terminal or not, and the player.
        '''
        current : Node = self
        is_terminal = False
        ply = deepcopy(player)
        
        while current.is_expanded and not is_terminal:
            self.pruner.tot_moves += 1
            current : Node = current.get_best_child()
            ply = 1 - ply
            self.game._board = deepcopy(current.state)
            is_terminal = self.game.check_winner() != -1
        return current, is_terminal, ply


    def get_child(self, func : Node.get_random_state, expandable_actions : np.ndarray[int], 
                  rollout_player : Literal[0, 1]) -> tuple['PrunedNodeDynamic', int]:
        '''
        Returns a new child `Node`, given the legal moves and the current state. This creates a new `Node` and
        stores it in the parent `Node`. It exploits the `get_random_state` function, so it needs its arguments
        as well, except for the `rollout_state` which is inferred from the `Node` object that calls this
        function.

        Arguments
        ----------
        func : function
            Function that returns a new state and action given the legal moves and the current state. In
            particular, `get_random_state` just returns the state and the action, while `get_child` creates
            a new object as well;
        
        expandable_actions : np.ndarray[int]
            Array of legal actions encoded as integers;
        
        rollout_player : int
            Player that will make the move.
        
        Returns
        -------
        out : tuple['Node', int]
            Tuple containing the new child `Node` and the action taken.
        '''
        child_state, action = func(expandable_actions=expandable_actions,
                                   rollout_player=rollout_player)

        child = PrunedNodeDynamic(self.game, child_state, self.args, 
                                  parent=self, action_taken=action, helper=self.helper, pruner=self.pruner)
        self.children[action] = child
        return child, action
    
    expand = dynamic_expand(Node.expand)
    
    def reset_state(self, state : np.ndarray[int, int], backup_tot_moves : int) -> None:
        '''
        Function used to reset the state of the game to the one passed as argument. It is used to reset the
        the total number of moves made so far, fundamental for the dynamic pruning strategy, and the pruner's
        parameter.

        Arguments
        ---------
        state : np.ndarray[int, int]
            Old state of the game, to which the current state will be set. It is most likely the state
            of the root;
        
        backup_tot_moves : int
            The old number of moves made so far. It is used only in the dynamic pruning strategy to restore the
            value of the total moves.
        '''
        super().reset_state(state)
        self.pruner.tot_moves = deepcopy(backup_tot_moves)
        self.pruner.param = self.pruner.starting_param


class NeuralNodeStandard(Node):
    def __init__(self, game : Game, state : np.ndarray[int, int], args : dict, helper : Helper, 
                 parent : Union['StandardNode', None]=None, action_taken : int=None, model=None, prior_likelihood=0):
        '''
        Class representing a `Node` for the NeuralMCTS tree. It inherits from the `Node` class and
        implements the specific functionalities for the specific algorithm. In particular, it implements
        the functionalities to expand the tree using a neural network to predict both the policy and the value.

        Arguments
        ---------
        game : Game
            Instance of the game class;
        
        state : np.ndarray[int, int]
            Current state of the game;
        
        args : dict
            Dictionary with arguments for the MCTS algorithm, both general and variant-specific;
        
        helper : Helper
            Helper class that contains all the methods to encode/decode the game state and actions;
        
        parent : [StandardNode, None] = None
            Pointer to the parent `Node`, default is `None` for the root;
        
        action_taken : int
            Action taken by the player to reach the current state;
        
        model : torch.nn.Module
            Neural network model used to predict the policy and the value;
        
        prior_likelihood : float
            Prior likelihood of the action taken to reach the current state.
        '''
        super().__init__(game, state, args, helper, parent, action_taken)
        self.model = model
        self.prior_likelihood = prior_likelihood
        self.children_prior = np.zeros([48])
    
    @torch.no_grad()
    def expand(self, state : np.ndarray[int], policy : np.ndarray[float]) -> 'Node':
        '''
        Method to expand the current `Node` by using the policy predicted by the neural network. It creates
        a new child `Node` for each legal action and stores it in the parent `Node`.

        Arguments
        ---------
        state : np.ndarray[int, int]
            Current state of the game;
        
        policy : np.ndarray[float]
            Policy predicted by the neural network.
        
        Returns
        -------
        out : Node
            The new child `Node` created.
        '''
        backup_state = deepcopy(state)        
        
        # Generate a child for each legal action.
        for action, prob in enumerate(policy):
            if prob > 0:
                from_pos, encoded_move = self.helper.from_action_to_pos(action - 16)
                decoded_action = self.helper.decode_action(from_pos, encoded_move)

                # To play on the quixo board, we need to decode the action first.
                self.game._board = deepcopy(self.helper._from_code_to_quixo(backup_state))

                quixo_player = 0

                self.game.move(from_pos, decoded_action, quixo_player)                  # NOTE: make the move on the board, but now (0, 1).
                child_state = deepcopy(self.game.get_board())                           # Retrieve the board, but then turn into (-1, 1)
                child_state = deepcopy(self.helper._from_quixo_to_code(child_state))    # and finally flip to get the real version of the
                child_state = -child_state                                              # board. This becase the child state is the opposite
                                                                                        # of the parent.
                # We create a node with the (0, 1) format.
                child = NeuralNodeStandard(game=self.game, state=child_state, args=self.args, 
                                           parent=self, action_taken=action, helper=self.helper, 
                                           prior_likelihood=prob, model=self.model)
                self.children[action] = child
                self.children_visits[action] += 1
                self.children_prior[action] = prob
            self.game._board = deepcopy(backup_state)
     
        self.children_values += self._compute_UCB()
        self.is_expanded = True

    def get_best_child(self) -> 'Node':
        '''
        Returns the child of the current `Node` with the highest UCB value. This is used to select the
        most promising child `Node` during the selection phase.

        Returns
        -------
        out : Node
            The child `Node` with the highest UCB value.
        '''
        ucb_values = self._compute_UCB()
        if max(ucb_values) == 0:
            ucb_values[ucb_values == 0] = np.inf
            best_action = np.argmin(np.abs(ucb_values))
        else:
            best_action = np.argmax(ucb_values)
        self.children_visits[best_action] += 1
        return self.children[best_action]

    def select(self, player : Literal[0, 1]) -> tuple['Node', bool, int]:
        '''
        Starting from the root, select the best child according to the UCB formula until a leaf `Node` is
        reached.

        Arguments
        ---------
        player : Literal[0, 1]
            Player that will make the move before the expansion.
        
        Returns
        -------
        out : Node, bool, int
            The leaf `Node` reached, it could be either terminal or not, and the player.
        '''
        current : NeuralNodeStandard = self
        is_terminal = False
        ply = deepcopy(player)
        while current.is_expanded and not is_terminal:
            current : NeuralNodeStandard = current.get_best_child()
            ply = -ply
            self.game._board = deepcopy(self.helper._from_code_to_quixo(current.state))
            is_terminal = self.game.check_winner() != -1
        
        return current, is_terminal, ply
    
    def _compute_Q(self) -> np.ndarray[float]:
        '''
        Compute the Q value for each child `Node` as the probability of winning the game.

        Returns
        -------
        out : np.ndarray[float]
            Array of Q values for each child `Node`.
        '''
        mask_zero = self.children_visits == 0       # Mask out illegal moves.
        q_array = np.zeros([48], dtype=np.float64)
        if np.sum(~mask_zero) > 0:
            q_array[~mask_zero] = self.children_values[~mask_zero] / self.children_visits[~mask_zero]
        return q_array    

    def _compute_U(self) -> np.ndarray[float]:
        '''
        Compute the U value for each child `Node` considering the total visits of the current `Node`,
        the local visits of each child and the prior likelihood of the action.

        Returns
        -------
        out : np.ndarray[float]
            Array of U values for each child `Node`.
        '''
        mask = self.children_visits > 0
        u_array = np.zeros([48], dtype=np.float64)
        u_array[mask] = self.args['c'] * np.sqrt(self.number_visits) / (self.children_visits[mask] + 1) * self.children_prior[mask]
        return u_array

    def _compute_UCB(self) -> np.ndarray[float]:
        '''
        Compute the UCB value for each child `Node` as the sum of the Q and U values.

        Returns
        -------
        out : np.ndarray[float]
            Array of UCB values for each child `Node`.
        '''
        return self._compute_Q() + self._compute_U()
    
    def backprop(self, value : int) -> None:
        '''
        Backpropagate the value up to the root starting from the expanded `Node`. This is done by updating
        the total value of the parent `Node` and then calling the `backprop` method on the parent.

        Arguments
        ---------
        value : int
            Value of the terminal state, representing the outcome of the game.
        '''
        try:
            self.total_value += value
            self.parent.backprop(-value)
        
        except AttributeError:  # This exception is raised when the root node is reached, since there will
            pass                # be no parent (self.parent is None).

    def simulate(self, rollout_player : Literal[0, 1]) -> int:
        raise NotImplementedError('This method is not implemented in the neural version of the MCTS.')
    
    def check_expandable_nodes(self) -> np.ndarray[int] | None:
        raise NotImplementedError('This method is not implemented in the neural version of the MCTS.')
    
    def get_child(self):
        raise NotImplementedError('This method is not implemented in the neural version of the MCTS.')
