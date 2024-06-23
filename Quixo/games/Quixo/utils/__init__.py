from games.Quixo.core.game import Move

import numpy as np

# This is called everytime the module is imported. Useful to avoid the overhead of reading the configuration
# file every time it is needed.


# XXX: This is a hack to avoid circular imports. The configuration file
# must be located in the specified directory. If using wandb, comment this.
# with open('games/Quixo/config_files/conf.yaml', 'rb') as f:         # * >>> -----------><><------------>> *
#     conf = yaml.safe_load(f.read())                                 # * MUST BE LOCATED IN THIS DIRECTORY *
#     MAX = conf['Pruner']['beta_param'] if 'Pruner' in conf else 1   # * <<------------><><----------- <<< *

#     # Sanity check to avoid cyclic games (i.e. games that can't end) during the simulation phase.
#     max_moves_simulation = conf['MCTS']['max_moves_simulation'] if ['max_moves_simulation'] in conf['MCTS'] else 250


# Remove this if not using wandb.

MAX = 50
max_moves_simulation = 300

legal_moves = np.array(
    [-15, -14, -13, -11, -10,  -9,  -7,  -6,  -5,  -3,  -2,  -1,   0,
       1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
      14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
      27,  28,  29,  30,  31])

action_pos = np.array(
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
         (1, 4), (2, 4), (3, 4), (4, 4),
         (4, 3), (4, 2), (4, 1), (4, 0),
         (3, 0), (2, 0), (1, 0)])

# Since each action refers to a relative position, we need to decode it
# into an absolute one.
decode_enum = {
            'first_row' : {
                Move.TOP : Move.BOTTOM,
                Move.RIGHT : Move.LEFT,
                Move.LEFT : Move.RIGHT
                },
            'right_column' : {
                Move.TOP : Move.LEFT,
                Move.RIGHT : Move.TOP,
                Move.LEFT : Move.BOTTOM
                },
            'left_column' : {
                Move.TOP : Move.RIGHT,
                Move.RIGHT : Move.BOTTOM,
                Move.LEFT : Move.TOP
                },
            'last_row' : {
                Move.TOP : Move.TOP,
                Move.RIGHT : Move.RIGHT,
                Move.LEFT : Move.LEFT
                },
            'first_corner' : {
                Move.TOP : Move.RIGHT,
                Move.RIGHT : Move.BOTTOM
                },
            'second_corner' : {
                Move.TOP : Move.BOTTOM,
                Move.RIGHT : Move.LEFT
                },
            'third_corner' : {
                Move.TOP : Move.LEFT,
                Move.RIGHT : Move.TOP
                },
            'fourth_corner' : {
                Move.TOP : Move.TOP,
                Move.RIGHT : Move.RIGHT
                }
        }