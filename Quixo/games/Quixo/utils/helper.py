import numpy as np

from games.Quixo.core.game import Game, Move
from copy import deepcopy, copy
from typing import Literal, Union
from . import legal_moves, action_pos, decode_enum, max_moves_simulation


class Helper():
    def __init__(self, game : Game) -> None:
        '''
        Class which acts as a support for the MCTS algorithm. It is used to encode and decode the actions
        that are performed during the search, and to check which moves are legal and which are not. It also
        provides a way to translate the actions from the algorithm perspective to the game class perspective.

        Arguments
        ---------
        game : Game
            The game class that is being played;
        
        Move : Move = Move
            The Move class that is being used. This should be fixed by default.
        '''
        self.game = game
        self.legal_moves = legal_moves                      # * >>> ----------><><---------->> *
        self.action_pos = action_pos                        # * IMPORTING FROM THE __init__.py *
        self.decode_enum = decode_enum                      # * <<----------><><---------- <<< *
        self._max_moves_simulation = max_moves_simulation

    @property
    def max_moves_simulation(self) -> int:
        return self._max_moves_simulation

    def _get_border(self, state : np.ndarray[int, int]) -> np.ndarray[int]:
        '''
        Returns the values that lie on the borders of the board. These are the only pieces
        that can be moved by the player, so it makes sense to check which of them are neutral
        or belong to each player.

        Arguments
        ---------
        state : np.ndarray[int, int]
            The current state of the game.
        
        Returns
        -------
        out : np.ndarray[int]
            The values that lie on the borders of the board.
        '''
        return np.array([state[i, j] for i, j in self.action_pos])
    
    def _get_playable_pieces(self, state : np.ndarray[int, int], 
                             player : Literal[-1, 1]) -> np.ndarray[bool]:
        '''
        Check which pieces can be moved by the player. These are the pieces that lie on the borders of
        the board and are either neutral or belong to the player.

        Arguments
        ---------
        state : np.ndarray[int, int]
            The current state of the game;
        
        player : int
            The player that is currently playing.
        
        Returns
        -------
        out : np.ndarray[bool]
            A binary array that tells which pieces can be moved by the player.
        '''
        # XXX: this works for standard, check for Neural.
        code_state = deepcopy(state)
        code_state = self._from_quixo_to_code(code_state)
        player = (-2 * player + 1)
        borders = self._get_border(code_state)
        neutral_or_player_pieces = np.logical_or(borders == player, borders == 0)
        return neutral_or_player_pieces
    
    def _from_quixo_to_code(self, state : np.ndarray[int, int]) -> np.ndarray[int, int]:
        return np.array(list(
                             map(lambda x: {1: -1, 0: 1, -1: 0}[x], state.reshape(-1))
                             )
                        ).reshape(state.shape)
    
    def _flip(self, state : np.ndarray[int, int], player : int=1) -> np.ndarray[int, int]:
        if player:
            return deepcopy(np.array(list(
                                map(lambda x: {-1: -1, 0: 1, 1: 0}[x], state.reshape(-1))
                                )
                            ).reshape(state.shape))
        return state
    
    def _from_quixo_to_neural(self, state : np.ndarray[int, int]) -> np.ndarray[int, int]:
        return np.array(list(
                             map(lambda x: {-1: 0, 0: 1, 1: -1}[x], state.reshape(-1))
                             )
                        ).reshape(state.shape)
    
    def _from_neural_to_quixo(self, state : np.ndarray[int, int]) -> np.ndarray[int, int]:
        return np.array(list(
                             map(lambda x: {0: -1, 1: 0, -1: 1}[x], state.reshape(-1))
                             )
                        ).reshape(state.shape)

    def _from_code_to_quixo(self, state : np.ndarray[int, int]) -> np.ndarray[int, int]:
        return np.array(list(
                             map(lambda x: {-1: 1, 1: 0, 0: -1}[x], state.reshape(-1))
                             )
                        ).reshape(state.shape)

    def _get_indices_playable_pieces(self, state : np.ndarray[int, int], player=-1) -> np.ndarray[int]:
        '''
        Returns the indices of the pieces that can be moved by the player. The indices are associated
        to the border of the board, so they are relative to the position of the pieces.

        Arguments
        ---------
        state : np.ndarray[int]
            The current state of the game.
        
        Returns
        ----------
        out : np.ndarray[int]
            The indices of the pieces that can be moved by the player because either neutral or
            belong to the player.
        '''
        # encoded_state = deepcopy(self._from_quixo_to_code(state))                   # Get encoded state.
        encoded_state = deepcopy(state)
        playable_pieces = self._get_playable_pieces(encoded_state, player=player)   # Get pieces that are either neutral or belong to the player.
        return np.argwhere(playable_pieces).reshape(-1)                             # Get the indices of the pieces that can be moved.
    
    def _get_every_playable_move(self, legal_pieces : np.ndarray[int]) -> tuple[np.ndarray[bool], np.ndarray[int]]:
        '''
        Starting from the indices of the pieces that can be moved by the player, this function
        returns every legal move that can be performed by the player, both in integer and binary
        format.

        Arguments
        ---------
        legal_pieces : np.ndarray[int]
            The indices of the pieces that can be moved by the player.
        
        Returns
        -------
        out : tuple[np.ndarray[bool], np.ndarray[int]]
            A tuple containing the binary and the integer positional representation of the legal moves.
        '''
        # Get all possible integer move and filter only the legal one
        all_candidate_moves = np.hstack((legal_pieces - 16, legal_pieces, legal_pieces + 16))                   
        legal_moves_for_current_player = all_candidate_moves[np.isin(all_candidate_moves, self.legal_moves)]

        # Get the binary and the integer positional representation of the legal moves.
        integer_legal_moves = legal_moves_for_current_player + 16
        binary_values_len_48 = np.isin(np.arange(48), integer_legal_moves)
        return binary_values_len_48, integer_legal_moves

    def get_legal_moves(self, state : np.ndarray[int, int], player=-1) -> tuple[np.ndarray[bool], np.ndarray[int]]:
        '''
        Returns all the legal moves that can be performed by the player, both in integer and binary
        format.

        Arguments
        ---------
        state : np.ndarray[int, int]
            The current state of the game.
        
        Returns
        -------
        out : tuple[np.ndarray[bool], np.ndarray[int]]
            A tuple containing the binary and the integer positional representation of the legal moves.
        '''
        legal_pieces = self._get_indices_playable_pieces(state, player=player)
        binary_values_len_48, integer_legal_moves = self._get_every_playable_move(legal_pieces)
        return binary_values_len_48, integer_legal_moves
    
    def from_action_to_pos(self, action : int) -> tuple[tuple[int, int], Move]:
        '''
        Decode the action from an integer format to a tuple format according to the game class.

        Arguments
        ---------
        action : int
            The action to decode.
        
        Returns
        -------
        out : tuple[tuple[int, int], Move]
            The decoded action.
        
        Note
        ----------
        This is possible thanks to the encoding used. If an action is negative, then it means that
        16 has been subtracted and so that the move is to the relative left of the piece. If greater
        than 16, then it means that 16 has been added and so that the move is to the relative right
        and, if between, then it means that the move is to the relative top.
        '''
        if action < 0:
            return self.action_pos[action + 16], Move.LEFT
        elif action > 15:
            return self.action_pos[action - 16], Move.RIGHT
        return self.action_pos[action], Move.TOP
    
    def from_pos_to_action(self, from_pos : tuple[int, int], move : Move) -> int:
        '''
        Encode an action from the tuple format into the integer one. Notice this function
        only returns a number, not something that can be used to move the piece.

        Arguments
        ---------
        from_pos : tuple[int, int]
            The position of the piece to move;
        
        move : Move
            The direction in which the piece has to be moved, considering the absolute position.
        
        Returns
        -------
        out : int
            The encoded action in integer format.
        
        Note
        ----
        Since the move is assumed to be provided in an absolute way, and not relative to the piece,
        it is mandatory to define a mapping from absolute to relative.
        '''
        if move == Move.LEFT:
            return np.argwhere(np.all(self.action_pos == from_pos, axis=1))
        elif move == Move.RIGHT:
            return np.argwhere(np.all(self.action_pos == from_pos, axis=1)) + 32
        return np.argwhere(np.all(self.action_pos == from_pos, axis=1)) + 16

    def decode_action(self, from_pos : tuple[int, int], action : Move) -> Move:
        '''
        Starting from a position and a relative move, returns the absolute move that can be applied.
        This adheres to the game class logic and acts as a translator from the algorithm perspective.

        Arguments
        ---------
        from_pos : tuple[int, int]
            The position of the piece to move.
        
        action : Move
            The relative move to perform, according to the algorithm.
        
        Returns
        -------
        out : Move
            The absolute move to perform, according to the game class.
        
        Note
        ----
        The different actions are encoded considering the following logic:
        >>> # > v v v v
        >>> # >       <
        >>> # >       <
        >>> # >       <
        >>> # ^ ^ ^ ^ <
        '''
        if from_pos[0] == 0 and from_pos[1] != 0:
            return self.decode_enum['first_row'][action]
        elif from_pos[1] == 4 and from_pos[0] != 0:
            return self.decode_enum['right_column'][action]
        elif ((from_pos[0] < 4) and (from_pos[0] != 0)) and (from_pos[1] == 0):
            return self.decode_enum['left_column'][action]
        elif ((from_pos[1] < 4) and (from_pos[1] != 0)) and (from_pos[0] == 4):
            return self.decode_enum['last_row'][action]
        elif from_pos[0] == 0 and from_pos[1] == 0:
            return self.decode_enum['first_corner'][action]
        elif from_pos[0] == 0 and from_pos[1] == 4:
            return self.decode_enum['second_corner'][action]
        elif from_pos[0] == 4 and from_pos[1] == 4:
            return self.decode_enum['third_corner'][action]
        elif from_pos[0] == 4 and from_pos[1] == 0:
            return self.decode_enum['fourth_corner'][action]
        return action
    
    def from_game_to_tensor(self, state):
        return np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(int)
    
    # # FIXME: seems this function is never called.
    # def get_next_state(self, state, action, player):
    #     next_state = deepcopy(state)
    #     from_pos, encoded_slide = self.from_action_to_pos(action - 16)
    #     decoded_slide = self.decode_action(from_pos, encoded_slide)
    #     self.__move(from_pos, decoded_slide, player)

    # def play(self, player1: Player, player2: Player) -> int:
    #     '''Play the game. Returns the winning player'''
    #     players = [player1, player2]
    #     winner = -1
    #     while winner < 0:
    #         self.current_player_idx += 1
    #         self.current_player_idx %= len(players)
    #         ok = False
    #         while not ok:
    #             # The player must always return a tuple + slide, so the action decoding
    #             # must happen within the make_move method.
    #             from_pos, slide = players[self.current_player_idx].make_move(
    #                 self)
    #             ok = self.__move(from_pos, slide, self.current_player_idx)
    #         winner = self.check_winner()
    #     return winner

    # def __move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
    #     '''Perform a move'''
    #     if player_id > 2:
    #         return False
        
    #     # Oh God, Numpy arrays
    #     prev_value = deepcopy(self._board[(from_pos[1], from_pos[0])])
    #     acceptable = self.__take((from_pos[1], from_pos[0]), player_id)
    #     if acceptable:
    #         acceptable = self.__slide((from_pos[1], from_pos[0]), slide)
    #         if not acceptable:
    #             self._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
    #     return acceptable

    # def __take(self, from_pos: tuple[int, int], player_id: int) -> bool:
    #     '''Take piece'''
    #     # acceptable only if in border
    #     acceptable: bool = (
    #         # check if it is in the first row
    #         (from_pos[0] == 0 and from_pos[1] < 5)
    #         # check if it is in the last row
    #         or (from_pos[0] == 4 and from_pos[1] < 5)
    #         # check if it is in the first column
    #         or (from_pos[1] == 0 and from_pos[0] < 5)
    #         # check if it is in the last column
    #         or (from_pos[1] == 4 and from_pos[0] < 5)
    #         # and check if the piece can be moved by the current player
    #     ) and (self._board[from_pos] < 0 or self._board[from_pos] == player_id)
    #     if acceptable:
    #         self._board[from_pos] = player_id
    #     return acceptable

    # def __slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
    #     '''Slide the other pieces'''
    #     # define the corners
    #     SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
    #     # if the piece position is not in a corner
    #     if from_pos not in SIDES:
    #         # if it is at the TOP, it can be moved down, left or right
    #         acceptable_top: bool = from_pos[0] == 0 and (
    #             slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
    #         )
    #         # if it is at the BOTTOM, it can be moved up, left or right
    #         acceptable_bottom: bool = from_pos[0] == 4 and (
    #             slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
    #         )
    #         # if it is on the LEFT, it can be moved up, down or right
    #         acceptable_left: bool = from_pos[1] == 0 and (
    #             slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
    #         )
    #         # if it is on the RIGHT, it can be moved up, down or left
    #         acceptable_right: bool = from_pos[1] == 4 and (
    #             slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
    #         )
    #     # if the piece position is in a corner
    #     else:
    #         # if it is in the upper left corner, it can be moved to the right and down
    #         acceptable_top: bool = from_pos == (0, 0) and (
    #             slide == Move.BOTTOM or slide == Move.RIGHT)
    #         # if it is in the lower left corner, it can be moved to the right and up
    #         acceptable_left: bool = from_pos == (4, 0) and (
    #             slide == Move.TOP or slide == Move.RIGHT)
    #         # if it is in the upper right corner, it can be moved to the left and down
    #         acceptable_right: bool = from_pos == (0, 4) and (
    #             slide == Move.BOTTOM or slide == Move.LEFT)
    #         # if it is in the lower right corner, it can be moved to the left and up
    #         acceptable_bottom: bool = from_pos == (4, 4) and (
    #             slide == Move.TOP or slide == Move.LEFT)
    #     # check if the move is acceptable
    #     acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
    #     # if it is
    #     if acceptable:
    #         # take the piece
    #         piece = self._board[from_pos]
    #         # if the player wants to slide it to the left
    #         if slide == Move.LEFT:
    #             # for each column starting from the column of the piece and moving to the left
    #             for i in range(from_pos[1], 0, -1):
    #                 # copy the value contained in the same row and the previous column
    #                 self._board[(from_pos[0], i)] = self._board[(
    #                     from_pos[0], i - 1)]
    #             # move the piece to the left
    #             self._board[(from_pos[0], 0)] = piece
    #         # if the player wants to slide it to the right
    #         elif slide == Move.RIGHT:
    #             # for each column starting from the column of the piece and moving to the right
    #             for i in range(from_pos[1], self._board.shape[1] - 1, 1):
    #                 # copy the value contained in the same row and the following column
    #                 self._board[(from_pos[0], i)] = self._board[(
    #                     from_pos[0], i + 1)]
    #             # move the piece to the right
    #             self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
    #         # if the player wants to slide it upward
    #         elif slide == Move.TOP:
    #             # for each row starting from the row of the piece and going upward
    #             for i in range(from_pos[0], 0, -1):
    #                 # copy the value contained in the same column and the previous row
    #                 self._board[(i, from_pos[1])] = self._board[(
    #                     i - 1, from_pos[1])]
    #             # move the piece up
    #             self._board[(0, from_pos[1])] = piece
    #         # if the player wants to slide it downward
    #         elif slide == Move.BOTTOM:
    #             # for each row starting from the row of the piece and going downward
    #             for i in range(from_pos[0], self._board.shape[0] - 1, 1):
    #                 # copy the value contained in the same column and the following row
    #                 self._board[(i, from_pos[1])] = self._board[(
    #                     i + 1, from_pos[1])]
    #             # move the piece down
    #             self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
    #     return acceptable
    

# class Buffer(ABC):

#     @abstractmethod
#     def store(self) -> None:
#         pass

#     @abstractmethod
#     def get_node_from_action(self):
#         pass


class ReuseBuffer:
    def __init__(self, helper : Helper) -> None:
        '''
        An object of this class is used to store the nodes that are generated during the
        MCTS search. The children of the root are stored in a dictionary where the key is 
        the action that led to the node and the value is the node itself. This is useful to
        reuse the nodes that have already been generated during the search, and make the
        search more efficient by avoiding the generation of the same nodes multiple times,
        not wasting all the necessary statistics.
        '''
        self._helper = helper
    
    def store(self, children : np.ndarray[object]) -> None:
        '''
        Link a child to the action that led to it.

        Arguments
        ---------
        children : list
            The list of children to store for possible reuse.
        '''
        self._buffer = copy(children)
    
    def get_node_from_action(self, from_pos : tuple, action : Move) -> Union[object, None]:
        '''
        Get the node that was generated by the action taken.

        Arguments
        ---------
        from_pos : tuple
            The position of the piece that was moved;

        action : int
            The action that led to the node.
        
        Returns
        -------
        out : object
            The node that was generated by the action taken.
        '''
        
        # The action of the opponent must be first encoded as an action, and then put into
        # the dictionary.
        decoded_action = self._helper.from_pos_to_action(from_pos, action)
        try:
            return self._buffer[decoded_action]
        except AttributeError:
            return None

class NeuralWrapper:
    def __init__(self, game):
        self.game = game

    def from_game_to_tensor(self, state):
        return np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(int)