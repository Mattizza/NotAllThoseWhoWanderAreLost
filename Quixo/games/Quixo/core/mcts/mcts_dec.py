import numpy as np
import torch

from . import TERMINAL_VALUES_MAPPING
from games.Quixo.core.game import Game
from games.Quixo.utils.helper import ReuseBuffer
from games.Quixo.utils.pruner import Pruner
from games.Quixo.core.mcts.nodes import StandardNode, PrunedNodeStatic, PrunedNodeDynamic, NeuralNodeStandard, Node
from copy import deepcopy, copy
from abc import ABC, abstractmethod


class MCTS(ABC):
    def __init__(self, game : Game, args : dict, buffer : ReuseBuffer=None) -> None:
        '''
        This class implements the Monte Carlo Tree Search algorithm. It has only one method, `gamble`, that
        takes the current state of the game and returns the probabilities of each action to be taken. The
        probabilities are calculated as the ratio between the number of visits of each child node and the total
        number of visits of the root node. Intuitively, the more a node is visited, the more likely it is to be
        selected according to the formulas adopted.

        Arguments
        ---------
        game : Game
            Instance of a `Game` class;
        
        args : dict
            Dictionary with arguments for the MCTS algorithm;
         
        buffer : ReuseBuffer = None
            Buffer used to store the states of the game and avoid recomputing them.
        '''
        self.game = game    
        self.args = args
        self.buffer = buffer            # We create just one Helper object, and we pass it through the nodes. We
        self.helper = buffer._helper    # don't create a new object for each node to avoid overheads.

    def gamble_loop(self, root : StandardNode | PrunedNodeStatic | PrunedNodeDynamic, backup_state : np.ndarray[int, int], 
                    backup_tot_moves : int=None) -> tuple[Node, Node]:
        '''
        This method is used to simulate the MCTS algorithm. It implements a loop that selects a `Node`, expands it,
        simulates the game, and backpropagates the value. The loop is repeated for a number of times equal to the
        number of searches specified in the arguments.

        Arguments
        ---------
        root : StandardNode | PrunedNodeStatic | PrunedNodeDynamic
            The root `Node` of the tree;
        
        backup_state : np.ndarray[int, int]
            The state of the game at the beginning of the search;
        
        backup_tot_moves : int = None
            The number of moves made so far. It is used only in the dynamic pruning strategy to restore the
            value of the total moves.
        
        Returns
        -------
        out : tuple[Node, Node]
            The last `Node` selected and the root `Node`.
        '''

        for _ in range(self.args['num_searches']):
            node : Node = root
            # Reset the state to the actual one. This because, when simulating, the game board is changed inplace.
            node.reset_state(backup_state, backup_tot_moves)
            # The Agent always thinks he is player 1, so we need to adopt its perspective at the beginning.
            player = 1
                 
            # If a terminal node is selected, skip expansion and simulation and go directly to backpropagation.
            # Otherwise, select a non-expanded node.
            child, is_terminal, player = node.select(player)
            value, _ = TERMINAL_VALUES_MAPPING[self.game.check_winner()]

            try:
                # If not terminal, expand the node and simulate the game.
                assert not is_terminal
                child = child.expand(player)
                value, is_terminal = TERMINAL_VALUES_MAPPING[self.game.check_winner()]

                
                # Once expanded, switch the player.
                player = 1 - player
                try:
                    # Check again whether terminal once expanded.
                    assert not is_terminal                              # Coincise way to map {0 : 1, 1 : -1}.
                    value = (-2 * player + 1) * child.simulate(player)  # Depending on which player won, and depending on which was
                    child.backprop(value)                               # the player from which the simulation started, we need
                                                                        # to flip the sign accordingly, penalizing or not.
                except AssertionError:
                    value = (-2 * player + 1) * value
                    child.backprop(value)           
            except AssertionError:
                value = (-2 * player + 1) * value
                child.backprop(value)
        return node, root
    
    @abstractmethod
    def gamble():
        '''
        This method should wrap `gamble_loop` and be used to implement the MCTS algorithm. It should add some
        extra functionalities to the loop, depending on the variant of the algorithm we want to implement.
        '''
        pass

    @staticmethod
    def get_best_action(root : StandardNode | PrunedNodeStatic | PrunedNodeDynamic) -> tuple[int, np.ndarray[object]]:
        '''
        Static method that, given the root `Node`, returns the action with the highest probability to be selected
        and the children of the selected `Node`.

        Arguments
        ---------
        root : StandardNode | PrunedNodeStatic | PrunedNodeDynamic
            The root node of the tree.
        
        Returns
        -------
        out : tuple[int, np.ndarray[Node]]
            The action to be selected and the children of the selected `Node`.
        '''
        action_probs = np.zeros(48)
        action_probs += root.children_visits
        action_probs /= np.sum(action_probs)

        action = np.argmax(action_probs)                # Select the action with the highest probability.
        next_move_node = root.children[action]          # Get the node associated with the selected action.
        opponent_next_moves = next_move_node.children   # Get the children of the selected node, i.e. all 
                                                        # the possible states of the next iteration and store them.
        return action, opponent_next_moves

def standard(func : MCTS.gamble_loop) -> tuple[int, np.ndarray[object]]:
    '''
    Decorator around the `gamble_loop` method of the parent MCTS class. It is used to implement the standard
    variant, where the number of actions to explore is not reduced.

    Arguments
    ---------
    func : MCTS.gamble_loop
        The method to be decorated.
    
    Returns
    -------
    out : tuple[int, np.ndarray[object]]
        The action to be selected and the children of the selected node.
    '''
    def wrapper(self, state : np.ndarray[int, int], explored_root : StandardNode):
        # Store the state of the game at the beginning of the search. It will be used to reset the state
        # during the simulations.
        backup_state = deepcopy(state)
        # If the node has already been explored in previous iterations, we don't need to create a new one.
        if explored_root is None:
            root = StandardNode(game=self.game, state=state, args=self.args, helper=self.helper)
        else:
            root = copy(explored_root)
        backup_state = deepcopy(root.state)
        
        # Gamble loop.
        node, root = func(self, root, backup_state)
        node : StandardNode

        # Need to reset also after last simulation, otherwise the game state will be changed.
        node.reset_state(state=backup_state)
        action, opponent_next_moves = MCTS.get_best_action(root)
        return action, opponent_next_moves
    return wrapper

def static(func : MCTS.gamble_loop) -> tuple[int, np.ndarray[object]]:
    '''
    Decorator around the `gamble_loop` method of the parent MCTS class. It is used to implement the pruned
    static variant, where the number of actions to explore is reduced, but considering a constant number of
    steps as going deeper in the tree.

    Arguments
    ---------
    func : MCTS.gamble_loop
        The method to be decorated.

    Returns
    -------
    out : tuple[int, np.ndarray[object]]
        The action to be selected and the children of the selected node.
    '''
    def wrapper(self, state : np.ndarray[int, int], explored_root : PrunedNodeStatic):
        backup_state = deepcopy(state)
        
        if explored_root is None:
            root = PrunedNodeStatic(game=self.game, state=state, args=self.args, helper=self.helper, pruner=self.pruner)
        else:
            root = copy(explored_root)
        node, root = func(self, root, backup_state)
        node : PrunedNodeStatic

        node.reset_state(state=backup_state)
        node.pruner.param = deepcopy(node.pruner.starting_param)

        action, opponent_next_moves = MCTS.get_best_action(node)
        
        # Increase by two because are considered the moves made both by the Agent and the Opponent.
        self.pruner.tot_moves += 2

        # NOTE: update the param after the move has been done, so that the next time we will have the correct
        # number of moves. This update occurs only after the effective move has been made, not during the
        # simulations.
        self.pruner._update_param()
        return action, opponent_next_moves
    return wrapper

def dynamic(func : MCTS.gamble_loop) -> tuple[int, np.ndarray[object]]:
    '''
    Decorator around the `gamble_loop` method of the parent MCTS class. It is used to implement the pruned
    dynamic variant, where the number of actions to explore is reduced, but considering a number of steps
    proportional to the number of moves made so far.

    Arguments
    ---------
    func : MCTS.gamble_loop
        The method to be decorated.
    
    Returns
    -------
    out : tuple[int, np.ndarray[object]]
        The action to be selected and the children of the selected node.
    '''
    def wrapper(self, state : np.ndarray[int, int], explored_root : PrunedNodeDynamic):
        backup_state = deepcopy(state)
        backup_tot_moves = deepcopy(self.pruner.tot_moves)

        if explored_root is None:
            root = PrunedNodeDynamic(game=self.game, state=state, args=self.args, helper=self.helper, pruner=self.pruner)
        else:
            root = copy(explored_root)

        # In this case we need to pass also the starting number of moves.
        node, root = func(self, root, backup_state, backup_tot_moves)
        node : PrunedNodeDynamic
        
        node.reset_state(state=backup_state, backup_tot_moves=backup_tot_moves)

        # FIXME: check this, may be redundant.
        node.pruner.param = deepcopy(node.pruner.starting_param)
        
        action, opponent_next_moves = MCTS.get_best_action(node)
    
        self.pruner.tot_moves += 2
        self.pruner._update_param()

        return action, opponent_next_moves
    return wrapper


class StandardMCTS(MCTS):
    def __init__(self, game : Game, args : dict, buffer : ReuseBuffer=None,
                 pruner : Pruner=None) -> None:
        '''
        Standard implementation of the MCTS algorithm, the most basic one. It inherits the `gamble_loop`
        method from the parent class, decorating it with the `standard` decorator, and it implements the
        `gamble` method.

        Arguments
        ---------
        game : Game
            Instance of a Game class;
        
        args : dict
            Dictionary with arguments for the MCTS algorithm;
        
        buffer : ReuseBuffer = None
            Buffer used to store the states of the game and avoid recomputing them.
        
        pruner : Pruner = None
            Instance of the Pruner class, set to `None` by default since not used in this variant but needed
            for the others.
        '''
        super().__init__(game, args, buffer)
    
    gamble = standard(MCTS.gamble_loop)


class PrunedMCTSStatic(MCTS):
    def __init__(self, game : Game, args : dict, buffer : ReuseBuffer=None,
                 pruner : Pruner=None) -> None:
        '''
        Subclass of the MCTS class, used to implement the Monte Carlo Trim Search algorithm. It inherits the
        `gamble_loop` method from the parent class, decorating it with the `static` decorator, and it implements
        the `gamble` method. This class also makes use of the `Pruner` class.

        Arguments
        ---------
        game : Game
            Instance of a game class;
        
        args : dict
            Dictionary with arguments for the MCTS algorithm;
        
        buffer : ReuseBuffer = None
            Buffer used to store the states of the game and avoid recomputing them;
        
        pruner : Pruner = None
            Instance of the Pruner class.
        '''
        super().__init__(game, args, buffer)
        self.pruner = pruner

    gamble = static(MCTS.gamble_loop)


class PrunedMCTSDynamic(MCTS):
    def __init__(self, game : Game, args : dict, buffer : ReuseBuffer=None,
                 pruner : Pruner=None) -> None:
        '''
        Subclass of the MCTS class, used to implement the Monte Carlo Trim Search algorithm. It inherits the
        `gamble_loop` method from the parent class, decorating it with the `dynamic` decorator, and it implements
        the `gamble` method. This class also makes use of the `Pruner` class.

        Arguments
        ---------
        game : Game
            Instance of a game class;
        
        args : dict
            Dictionary with arguments for the MCTS algorithm;
        
        buffer : ReuseBuffer = None
            Buffer used to store the states of the game and avoid recomputing them;
        
        pruner : Pruner = None
            Instance of the Pruner class.
        '''
        super().__init__(game, args, buffer)
        self.pruner = pruner
    
    gamble = dynamic(MCTS.gamble_loop)


class NeuralMCTS(MCTS):
    def __init__(self, game : Game, args : dict, buffer : ReuseBuffer=None,
                 model=None) -> None:
        '''
        Subclass of the MCTS class, used to implement the Monte Carlo Tree Search algorithm with a neural network
        as the model. The `gamble` method is implemented to simulate the MCTS algorithm using the neural network
        to predict the policy and the value of the states.

        Arguments
        ---------
        game : Game
            Instance of a Game class;
        
        args : dict
            Dictionary with arguments for the MCTS algorithm when considering a neural network;
        
        buffer : ReuseBuffer = None
            Buffer used to store the states of the game and avoid recomputing them, not used in this variant;
        
        model : nn.Module
            Neural network model used to predict the policy and the value of the states.
        '''
        super().__init__(game, args, buffer)
        self.buffer = buffer
        self.helper = buffer._helper
        self.model = model


    @torch.no_grad()
    def gamble_loop(self, root : NeuralNodeStandard, backup_state : np.ndarray[int, int], 
                    backup_tot_moves : int=None, old_player=None) -> tuple[Node, Node]:
        '''
        This method is used to simulate the MCTS algorithm using a neural network as the model. It implements a loop
        that selects a `Node`, expands it, skips the simulation, and backpropagates the value. The loop is repeated
        for a number of times equal to the number of searches specified in the arguments.

        Arguments
        ---------
        root : NeuralNodeStandard
            The root `Node` of the tree;
        
        backup_state : np.ndarray[int, int]
            The state of the game at the beginning of the search;
        
        backup_tot_moves : int = None
            The number of moves made so far. It is used only in the dynamic pruning strategy to restore the
            value of the total moves.
        
        old_player : int
            The player that made the last move. It is used to flip the sign of the value and the board each time
            the loop is executed.
        '''


        # When doing the loop, the model should always think it is the one playing. So, if we want player 1 to win, we need to
        # flip the sign of the value and the board each time.
        for it in range(self.args['num_searches']):

            # Uncomment for debugging.
            # if (it + 1) % 100 == 0:
            #      print(f'Search: {it + 1}')

            player = deepcopy(old_player)           # First iteration player=1
            node : NeuralNodeStandard = copy(root)  # Node
    
            node.reset_state(state=backup_state, backup_tot_moves=backup_tot_moves)

            child, is_terminal, player = node.select(player)

            # Since the children has the board opposite to the parent, if the child wins without making any action, 
            # it is actually the parent that is winning.
            value, _ = TERMINAL_VALUES_MAPPING[self.game.check_winner()]
            termination_value = -value
            
            try:
                assert not is_terminal
                policy, value = self.model(
                    torch.tensor(self.helper.from_game_to_tensor(child.state), device=self.model.device, dtype=torch.float32).unsqueeze(0)
                )

                # TODO: check this, eventually can be made more efficient.
                # Generate a legal probability distribution.
                policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
                legal_moves, _ = self.helper.get_legal_moves(self.helper._from_code_to_quixo(child.state), player=player)
                policy *= legal_moves
                policy /= np.sum(policy)

                alpha_value = value.item()

                # Player is now player=-1 with state flipped.
                child.expand(child.state, policy)

                self.game._board = self.helper._from_code_to_quixo(child.state)
                child.backprop(alpha_value)
                
            except AssertionError:
                child.backprop(-termination_value)
                continue
        return node, root
    
    @torch.no_grad()
    def gamble(self, state, explored_root=None, neural_player=None) -> np.ndarray[float]:
        '''
        This method is used to implement the MCTS algorithm using a neural network as the model. It wraps the
        `gamble_loop` method and adds some extra functionalities to it.

        Arguments
        ---------
        state : np.ndarray[int, int]
            The current state of the game;
        
        explored_root : NeuralNodeStandard = None
            The root `Node` of the tree if already explored in previous iterations, not used in this variant;

        neural_player : int
            The player that made the last move. It is used to flip the sign of the value and the board each time
            the loop is executed.

        '''

        # TODO: consider passing the node and consider it as the root, backup state is player : (1, -1)
        canonical_state = deepcopy(state * neural_player)

        root = NeuralNodeStandard(game=self.game, state=canonical_state, args=self.args, helper=self.helper)
        backup_state = deepcopy(root.state)

        policy, _ = self.model(
            torch.tensor(self.helper.from_game_to_tensor(backup_state), dtype=torch.float32, device=self.model.device).unsqueeze(0)
        )
                
        policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

        state_q = self.helper._from_code_to_quixo(backup_state)
        valid_moves, _ = self.helper.get_legal_moves(state=state_q, player=1)
        
        # Apply random noise to the policy, only when training.
        # policy = (1 - self.args['dirichlet_eps']) * policy + self.args['dirichlet_eps'] * np.random.dirichlet([self.args['dirichlet_eps']] * 48)
        policy *= valid_moves
        policy /= np.sum(policy)

        # Generate all possible children and assign them the values of the policy.
        root.expand(policy=policy, state=backup_state)

        node, root = self.gamble_loop(root=root, backup_state=backup_state, old_player=neural_player) # Gamble loop.
        node : StandardNode

        # Need to reset also after last simulation, otherwise the game state will be changed.
        node.reset_state(state=backup_state)
    
        action_probs = np.zeros(48)
        action_probs += node.children_visits
        action_probs /= np.sum(action_probs) 
        return action_probs