import random
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import pickle as pkl

from typing import Literal
from games.Quixo.core.game import Game, Move, Player
from games.Quixo.utils.helper import Helper, ReuseBuffer
from copy import copy, deepcopy
from games.Quixo.utils.pruner import Pruner
from games.Quixo.core.mcts.mcts_dec import PrunedMCTSDynamic, StandardMCTS, PrunedMCTSStatic, NeuralMCTS
from games.Quixo.core.mcts.__init__ import TERMINAL_VALUES_MAPPING


class RandomPlayer(Player):
    def __init__(self, game : Game, player : int=0, selfplay : bool=False) -> None:
        '''
        Random player which selects a random move from a pool of legal moves.

        Arguments
        ---------
        game : Game
            The game object.
        
        player : int = 0
            The assigned player, set to 0 by default.
        
        selfplay : bool = False
            Whether the player is playing against itself or not. It is set to `False` by default.
        '''
        super().__init__()
        self.helper = Helper(game)
        self.player = player
        self.selfplay = selfplay
        
        self.__requires_mcts = False
    
    @property
    def requires_mcts(self) -> bool:
        return self.__requires_mcts

    def make_move(self, game : Game, from_pos : None, encoded_action : None,
                  player : Literal[0, 1]=0) -> tuple[tuple[int, int], Move]:
        '''
        Given the current state of the game, this method samples a random legal move and returns it.

        Arguments
        ---------
        game : Game
            The game object.
        
        player : Literal[0, 1] = 0
            The assigned player, set to 0 by default.
        
        Returns
        -------
        out : tuple[tuple[int, int], Move]
            The position of the piece to move and the move to make.
        '''     
        if not self.selfplay:
            player = self.player
        _, legal_moves = self.helper.get_legal_moves(game.get_board(), player=player)
        action = random.choice(legal_moves)

        from_pos, encoded_action = self.helper.from_action_to_pos(action - 16)
        decoded_action = self.helper.decode_action(from_pos, encoded_action)
        return from_pos, decoded_action, encoded_action


class HumanPlayer(Player):
    def __init__(self) -> None:
        '''
        Player which asks the user for an input to make a move. The user needs to input the row and column of the piece
        to move and the move to make, the last taking value in the discrete set `{TOP, BOTTOM, LEFT, RIGHT}`.
        '''
        super().__init__()
        self.__requires_mcts = False

    @property
    def requires_mcts(self) -> bool:
        return self.__requires_mcts

    def make_move(self, game : Game, from_pos : None, encoded_action : None,
                  player : Literal[0, 1]=0) -> tuple[tuple[int, int], Move]:
        '''
        Given the current state of the game, this method samples a random legal move and returns it.

        Arguments
        ---------
        game : Game
            The game object.
        
        player : Literal[0, 1] = 0
            The assigned player, set to 0 by default.
        
        Returns
        -------
        out : tuple[tuple[int, int], Move]
            The position of the piece to move and the move to make.
        '''
        print(f'Current state of the board\n{game.get_board()}')
        from_pos = (int(input('Enter row: ')), int(input('Enter column: ')))
        player_move = input('Enter move: ')
        if player_move == 'TOP':
            return from_pos, Move.TOP
        elif player_move == 'BOTTOM':
            return from_pos, Move.BOTTOM
        elif player_move == 'LEFT':
            return from_pos, Move.LEFT
        elif player_move == 'RIGHT':
            return from_pos, Move.RIGHT
        else:
            raise ValueError("Invalid move")


class TheGambler(Player):
    def __init__(self, game : Game, input : str | dict, player : int = 1) -> None:
        '''
        Class which implements the MCTS algorithm to play the game, playing the role of the parent class for the different
        implementations. It initialize both a `Buffer` and a `Helper` object, which act as support structures for the
        algorithms, and implements the most naive `make_move` method.

        Arguments
        ---------
        game : Game
            The game object.
        input : str | dict
            The path to the YAML configuration file (single run) or a dict of parameters (wandb)
        
        player : int = 1
            The assigned player, set to 1 by default.
        '''
        super().__init__()
        self.helper = Helper(game)
        self.buffer = ReuseBuffer(self.helper)
        self.moves_performed = 0
        self.player = player
        self.player_backup = deepcopy(player)
        self.__requires_mcts = True
        
        # XXX: This is a hack to allow for either a single path to the configuration or several configurations.
        # Remove the try except block in the final version because eventually slower.
        # with open(path, 'rb') as f:               KEEP THIS
        #     self.conf = yaml.safe_load(f.read())
        # self.args = self.conf['MCTS']
        try:
            with open(input, 'rb') as f:                # If a path is passed, then single run.
                self.conf = yaml.safe_load(f.read())
        except:
            assert isinstance(input, dict), "Invalid input type."
            self.conf = input                           # If a dict is passed, then wandb.
        self.args = self.conf

    @property
    def requires_mcts(self) -> bool:
        return self.__requires_mcts

    def make_move(self, game : Game, from_pos : tuple[int, int], encoded_action : int,
                  strategy : StandardMCTS | PrunedMCTSStatic | PrunedMCTSDynamic,
                  pruner : Pruner=None) -> tuple[tuple[int, int], Move]:
        '''
        Standard implementation of the MCTS algorithm. It initializes the MCTS object and starts the search from the
        current state of the game, finally returning a `tuple[tuple[int, int], Move]` containing both the location of
        the piece to move and where to move it.

        Arguments
        ---------
        game : Game
            The game object;
        
        from_pos : tuple[int, int]
            The position on the board from which the opponent made the move in the previous turn;
        
        encoded_action : int
            The encoded action of the opponent's move in the previous turn;

        strategy : StandardMCTS | PrunedMCTSStatic | PrunedMCTSDynamic
            The strategy to adopt when looking for the best action to perform;
        
        pruner : Pruner = None
            A `Pruner` acting as a support structure. It is used only when the strategy is not
            `StandardMCTS`, but its presence in the signature of the function is still required.

        Returns
        -------
        out : tuple[tuple[int, int], Move]
            The position of the piece to move and the move to make. 
        '''
        mcts = strategy(game, self.args, buffer=self.buffer, pruner=pruner)
        explored_root = copy(self.buffer.get_node_from_action(from_pos, encoded_action)) # Get the node where to start the MCTS

        # Whenever the Agent is playing as player 0, the board has to be flipped because it always thinks it is player 1.
        try:
            explored_root = explored_root[0][0]     # If the buffer exists, we can use it to start the MCTS;
            state = deepcopy(explored_root.state)   # otherwise, it will be None and the MCTS will start as usual.
        except:
            state = game.get_board()
            # state = self.helper._flip(state, self.player_backup)

        action, opponent_next_moves = mcts.gamble(state, explored_root=explored_root)
        # game._board = self.helper._flip(state, self.player_backup)
        
        # XXX: keep only when playing against random, otherwise += 1.
        # self.moves_performed += 2 # Take into account the opponent's move as well.
        self.moves_performed += 1
        self.buffer.store(opponent_next_moves)
        from_pos, encoded_action = self.helper.from_action_to_pos(action - 16)  # I have spent around 4 hours to try to
        decoded_action = self.helper.decode_action(from_pos, encoded_action)    # understand why I was being returned 
        return from_pos, decoded_action, encoded_action                         # illegal actions; decode_action was needed.


class StandardMCTSPlayer(TheGambler):
    def __init__(self, game : Game, path : str, player : int=1) -> None:
        '''
        The simplest MCTS variant. It inherits everything from the `TheGambler` class and needs only an initialization
        of the hyperparameters of the algorithm.

        Arguments
        ---------
        game : Game
            The game object.
        
        path : str
            The path to the YAML configuration file.
        
        player : int = 1
            The assigned player, set to 1 by default.
        '''
        super().__init__(game, path, player)
        
    def make_move(self, game : Game, from_pos : tuple[int, int], encoded_action : int,
                  strategy=StandardMCTS) -> tuple[tuple[int, int], Move]:
        '''
        This is the equivalent of the `make_move` method in the `TheGambler` class. It returns the position of the piece
        to move and the move to make obtained from the most standard implementation of the MCTS algorithm.

        Arguments
        ---------
        game : Game
            The game object;
        
        from_pos : tuple[int, int]
            The position on the board from which the opponent made the move in the previous turn;
        
        encoded_action : int
            The encoded action of the opponent's move in the previous turn;
        
        strategy : StandardMCTS = StandardMCTS
            The strategy to adopt when looking for the best action to perform. It is set to `StandardMCTS` by default.
        
        Returns
        -------
        out : tuple[tuple[int, int], Move]
            The position of the piece to move and the move to make.
        '''
        return super().make_move(game, from_pos, encoded_action, strategy)


class PrunedPlayerStatic(TheGambler):
    def __init__(self, game : Game, path : str, player : int=1) -> None:
        '''
        This variant implements a static pruning strategy. It inherits everything from the `TheGambler` class, it
        initializes a `Pruner` and needs only an initialization of the hyperparameters of the algorithm.

        Arguments
        ---------
        game : Game
            The game object.
        
        path : str
            The path to the YAML configuration file.

        player : int = 1
            The assigned player, set to 1 by default.
        '''
        super().__init__(game, path, player)
        # Select only the parameters needed for the Pruner.
        
        pruner_dict = {k: v for k, v in self.conf.items() if k not in ['c', 'num_searches']}
        
        # Select the parameters needed for the growth function and pass them to the Pruner.
        gfunction_dict = {k: v for k, v in pruner_dict.items() if k not in ['beta_param', 'temperature', 'starting_percentage', 'gfunction']}
        pruner_dict['growth_function_params'] = gfunction_dict

        # Remove growth function parameters from the Pruner dictionary.
        pruner_dict = {k: v for k, v in pruner_dict.items() if k not in gfunction_dict.keys()}
        self.pruner = Pruner(**pruner_dict)
    
    def make_move(self, game : Game, from_pos : tuple[int, int], encoded_action : int,
                  strategy=PrunedMCTSStatic) -> tuple[tuple[int, int], Move]:
        '''
        This method is the equivalent of the `make_move` method in the `TheGambler` class, but it also exploits
        the static pruning strategy. As before, it returns the position of the piece to move and the move to make.

        Arguments
        ---------
        game : Game
            The game object;
        
        from_pos : tuple[int, int]
            The position on the board from which the opponent made the move in the previous turn;
        
        encoded_action : int
            The encoded action of the opponent's move in the previous turn;
        
        strategy : PrunedMCTSStatic = PrunedMCTSStatic
            The strategy to adopt when looking for the best action to perform. It is set to `PrunedMCTSStatic` by default.
        
        Returns
        -------
        out : tuple[tuple[int, int], Move]
            The position of the piece to move and the move to make.
        '''
        return super().make_move(game, from_pos, encoded_action, strategy, pruner=self.pruner)


class PrunedPlayerDynamic(TheGambler):
    def __init__(self, game : Game, path : str, player : int=1) -> None:
        '''
        This variant implements a dynamic pruning strategy. It inherits everything from the `TheGambler` class, it
        initializes a `Pruner` and needs only an initialization of the hyperparameters of the algorithm.

        Arguments
        ---------
        game : Game
            The game object.
        
        path : str
            The path to the YAML configuration file.
        
        player : int = 1
            The assigned player, set to 1 by default.
        '''
        super().__init__(game, path, player)
        # Select only the parameters needed for the Pruner.
        # <<< REMOVE THIS
        # with open(path, 'r') as f:
        #     hyp_grid = yaml.load(f, Loader=yaml.SafeLoader)
        # dict_hyp = {}
        # dict_hyp.update(hyp_grid['MCTS'])
        # dict_hyp.update(hyp_grid['Pruner'])
        # dict_hyp.update(hyp_grid['Growth_function'])
        # self.conf = deepcopy(dict_hyp)
        # REMOVE THIS >>>
        pruner_dict = {k: v for k, v in self.conf.items() if k not in ['c', 'num_searches']}
        
        # Select the parameters needed for the growth function and pass them to the Pruner.
        gfunction_dict = {k: v for k, v in pruner_dict.items() if k not in ['beta_param', 'temperature', 'starting_percentage', 'gfunction']}
        pruner_dict['growth_function_params'] = gfunction_dict

        # Remove growth function parameters from the Pruner dictionary.
        pruner_dict = {k: v for k, v in pruner_dict.items() if k not in gfunction_dict.keys()}
        self.pruner = Pruner(**pruner_dict)
    
    def make_move(self, game : Game, from_pos : tuple[int, int], encoded_action : int,
                  strategy=PrunedMCTSDynamic) -> tuple[tuple[int, int], Move]:
        '''
        This method is the equivalent of the `make_move` method in the `TheGambler` class, but it also exploits
        the dynamic pruning strategy. As before, it returns the position of the piece to move and the move to make.

        Arguments
        ---------
        game : Game
            The game object;
        
        from_pos : tuple[int, int]
            The position on the board from which the opponent made the move in the previous turn;
        
        encoded_action : int
            The encoded action of the opponent's move in the previous turn;
        
        strategy : PrunedMCTSDynamic = PrunedMCTSDynamic
            The strategy to adopt when looking for the best action to perform. It is set to `PrunedMCTSDynamic` by default.
        
        Returns
        -------
        out : tuple[tuple[int, int], Move]
            The position of the piece to move and the move to make.
        '''
        return super().make_move(game, from_pos, encoded_action, strategy, pruner=self.pruner)


class NeuralPlayerStandard(Player):
    def __init__(self, game : Game, args, model, optimizer, player=1) -> None:
        super().__init__()
        self.game = game
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)
        self.helper = Helper(game)
        self.buffer = ReuseBuffer(self.helper)
        self.NeuralMCTS = NeuralMCTS(game, args, self.buffer, model)
        self.moves_performed = 0
        self.__requires_mcts = True
        if player == 0:
            self.player = 1
        else:
            self.player = -1
        
    @property
    def requires_mcts(self) -> bool:
        return self.__requires_mcts

    def generate_quixo_boards(self, game : Game, n_moves = 15):
        game._board = np.ones((5, 5)) * -1
        lonely_man = RandomPlayer(game, player=1, selfplay=True)
        okay = False

        while not okay:
            player = 1
            for _ in range(n_moves):
                from_pos, slide = lonely_man.make_move(game, player=player)
                ok = game.move(from_pos, slide, player)
                player = 1 - player
            if game.check_winner() != -1:
                okay = False
                game._board = np.ones((5, 5)) * -1
                lonely_man = RandomPlayer(game, player=1, selfplay=True)
            else:
                okay = True
                
        print(f'Starting configuration: \n{game.get_board()}')
        return game.get_board()


    def self_play(self):
        
        # When self playing, initialize the memory to return back and initialize the first player. The board has
        # to be reset to the initial state.
        self.game = Game()
        replay_memory = []          # States already seen.
        quixo_player = 0            # First player to move.
        state = np.zeros((5, 5))    # Initial state of the game.

        # Start from a randomly initialized game state to help convergence.
        # state = self.helper._from_quixo_to_code(self.generate_quixo_boards(deepcopy(self.game), 2))
        self.game._board = self.helper._from_code_to_quixo(deepcopy(state))

        
        # XXX: check which kind of state to pass to the MCTS. Why passing to a neutral state? Should the
        # neural net always be player 1?
        while True:

            # Store the state of the board since the changes are done in place.
            back = deepcopy(self.game._board)
            canonic_state = self.helper._from_quixo_to_code(self.game._board)
            
            if quixo_player == 0:
                neural_player = 1
            elif quixo_player == 1:
                neural_player = -1

            # Get the probs and save everything in the memory.
            action_probs = self.NeuralMCTS.gamble(state=state, neural_player=neural_player)
            replay_memory.append([canonic_state, action_probs, neural_player])

            # Allow for the selection of suboptimal moves.
            temperature_action_probs = np.power(action_probs, 1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(range(48), p=temperature_action_probs) # XXX: check this. Should it start from 0 or 1?
            # action = np.argmax(action_probs)

            # Once sampled an action, decode it and update the state.
            from_pos, encoded_move = self.helper.from_action_to_pos(action - 16) # XXX: check this, should be okay now.
            decoded_move = self.helper.decode_action(from_pos, encoded_move)
            
            self.game._board = deepcopy(back)
            self.game.move(from_pos, decoded_move, player_id=quixo_player)
            print(f'Game board: \n{self.game.get_board()}')
            print(f'Neural player {neural_player} just moved.')
            state = self.helper._from_quixo_to_code(self.game.get_board())
            _, is_terminal = TERMINAL_VALUES_MAPPING[self.game.check_winner()]
            quixo_player = 1 - quixo_player
            
            # Play until a terminal state is reached.
            if is_terminal:

                # If terminal, update the results depending on the first player to move.
                return_memory = []
                
                # Since the net always thinks it is neural_player=1, quixo_player=0, we flip the state of the board accordingly.
                for memory_state, memory_probs, memory_player in replay_memory:
                    state_from_neural_net_perspective = memory_player * memory_state
                    
                    result = 1 if memory_player == neural_player else -1

                    # Finally, return the memory to be used for training.
                    return_memory.append(
                        [self.helper.from_game_to_tensor(state_from_neural_net_perspective),
                         memory_probs, 
                         result])
                return return_memory

    def _train(self, memory_buffer):

        # Always shuffle the memory buffer to not take into account the temporal correlation between the samples.
        random.shuffle(memory_buffer)
        losses = []
        # TODO: consider using a DataLoader to handle the memory buffer.
        # For each batch in the memory buffer, train the model.
        for batch_idx in range(0, len(memory_buffer), self.args['batch_size']):
            try:
                sample = memory_buffer[batch_idx:min(batch_idx + self.args['batch_size'], len(memory_buffer) - 1)]
            
                # Unzip the sample by transposing the values.
                states, probs, results = zip(*sample)
                states, probs, results = np.array(states), np.array(probs), np.array(results).reshape(-1, 1)

                states = torch.tensor(states, dtype=torch.float32, device=self.model.device)
                probs = torch.tensor(probs, dtype=torch.float32, device=self.model.device)
                results = torch.tensor(results, dtype=torch.float32, device=self.model.device)

                # Forward pass.
                policy, value = self.model(states)
                policy_loss = F.cross_entropy(policy, probs)
                value_loss = F.mse_loss(value, results)

                # Compute the loss.
                loss = policy_loss + value_loss
                print(loss)
                losses.append(loss.item())
                wandb.log({'loss': loss.item(),
                           'policy_loss': policy_loss.item(),
                           'value_loss': value_loss.item()})
                # Backward pass.
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
            
            except:
                pass
        
        self.scheduler.step(np.mean(losses))
        print(f'Current learning rate: {self.scheduler.get_last_lr()}')
        wandb.log({'learning_rate': self.scheduler.get_last_lr()[0],
                   'dataset_size': len(memory_buffer),
                   'mean_loss': np.mean(losses)})


    def train(self):

        # for iteration in range(self.args['num_iterations']):
            
            # replay_memory = []

            # # First build the replay memory by playing against itself and then train the model.
            # # Save a checkpoint, and then continue with the next iteration.
            # # XXX: eval mode for the model.
            # self.model.eval()
            # for _ in range(self.args['self_play_iterations']):
            #     print(f'\n\n---ITERATION {_ + 1}---\n')
            #     replay_memory += self.self_play()   # Add new observations to the replay memory.
            #     if (_ + 1) % 15 == 0:
            #         with open(f'/Users/mattia/Documents/University/Q-uixo/games/Quixo/core/alpha_zero/replay_memory/replay_memory_{_ + 1}.pkl', 'wb') as f:
            #             pkl.dump(replay_memory, f)
            #         del replay_memory
            #         replay_memory = []
            
            # break
            with open(f'/Users/mattia/Documents/University/Q-uixo/games/Quixo/core/alpha_zero/replay_memory/replay_memory_15.pkl', 'rb') as f:
                replay_memory = pkl.load(f)
            with open(f'/Users/mattia/Documents/University/Q-uixo/games/Quixo/core/alpha_zero/replay_memory/replay_memory_20.pkl', 'rb') as f:
                replay_memory += pkl.load(f)
            with open(f'/Users/mattia/Documents/University/Q-uixo/games/Quixo/core/alpha_zero/replay_memory/replay_memory_30.pkl', 'rb') as f:
                replay_memory += pkl.load(f)
            with open(f'/Users/mattia/Documents/University/Q-uixo/games/Quixo/core/alpha_zero/replay_memory/replay_memory_40.pkl', 'rb') as f:
                replay_memory += pkl.load(f)
            with open(f'/Users/mattia/Documents/University/Q-uixo/games/Quixo/core/alpha_zero/replay_memory/replay_memory_45.pkl', 'rb') as f:
                replay_memory += pkl.load(f)
            with open(f'/Users/mattia/Documents/University/Q-uixo/games/Quixo/core/alpha_zero/replay_memory/replay_memory_50.pkl', 'rb') as f:
                replay_memory += pkl.load(f)
                self.model.train()
                for _ in range(self.args['num_epochs']):
                    self._train(replay_memory)
                
                torch.save(self.model.state_dict(), f'/Users/mattia/Documents/University/Q-uixo/games/Quixo/core/alpha_zero/checkpoints/model_{8}.pt')
                torch.save(self.optimizer.state_dict(), f'/Users/mattia/Documents/University/Q-uixo/games/Quixo/core/alpha_zero/checkpoints/optimizer_{8}.pt')
        
    def make_move(self, game : Game, from_pos : tuple[int, int], slide : Move, 
                  neural_player = -1) -> tuple[tuple[int, int], Move]:
        
        # If the player is -1, switch the board so that it thinks it is 1.
        self.model.eval()
        neural_player  = self.player
        neural_state = self.helper._from_quixo_to_code(game.get_board()) * neural_player
        tensor_state = self.helper.from_game_to_tensor(neural_state)
        policy, _ = self.model(
            torch.tensor(tensor_state, dtype=torch.float32, device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

        # Since player 1, consider legal moves for player 1 always.
        state_q = self.helper._from_code_to_quixo(neural_state)

        # XXX: Neural always thinks he is player 0. Change this 
        valid_moves, _ = self.helper.get_legal_moves(state=state_q, player=0)
        
        # Apply random noise to the policy.
        # policy = (1 - self.args['dirichlet_eps']) * policy + self.args['dirichlet_eps'] * np.random.dirichlet([self.args['dirichlet_eps']] * 48)
        policy *= valid_moves
        policy /= np.sum(policy)
        
        temperature_action_probs = np.power(policy, 1 / self.args['temperature'])
        temperature_action_probs /= np.sum(temperature_action_probs)
        
        # When playing against strong players, do not add noise.
        action = np.random.choice(range(48), p=temperature_action_probs)
        action = np.argmax(policy)
        
        from_pos, encoded_action = self.helper.from_action_to_pos(action - 16)  # I have spent around 4 hours to try to
        decoded_action = self.helper.decode_action(from_pos, encoded_action)    # understand why I was being returned 
        return from_pos, decoded_action, encoded_action                         # illegal actions; decode_action was needed.