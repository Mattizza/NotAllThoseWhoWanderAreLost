import yaml
import time
import torch
import numpy as np

from games.Quixo.core.game import Game
from games.Quixo.core.agents.moes_tavern import RandomPlayer, HumanPlayer, StandardMCTSPlayer, PrunedPlayerStatic, PrunedPlayerDynamic, NeuralPlayerStandard
from games.Quixo.core.alpha_zero.model import NeuralNet
from argparse import ArgumentParser
from sklearn.model_selection import ParameterGrid

parser = ArgumentParser()
parser.add_argument("-s", "--strategy", help="choose one of the implemented strategies between 'random', 'human', 'standard', 'pruned_static', 'pruned_dynamic', 'neural'", 
                    type=str, default='standard', choices=['random', 'human', 'standard', 'pruned_static', 'pruned_dynamic', 'neural'], required=False)
parser.add_argument("-rr", "--royal_rumble", help="whether to play a royal rumble between the strategies", type=bool, default=False, required=False)
parser.add_argument("-w", "--wandb", help="whether to log the results to Weights & Biases", type=bool, default=False, required=False)
parser.add_argument("-wp", "--wandb_setup_path", help="path to the Weights & Biases configuration file", type=str, default=None, required=False)
parser.add_argument("-hg", "--hyp_grid", help="path to the hyperparameter grid", type=str, default=None, required=False)
parser.add_argument("-hp", "--hyperparameters", help="path to the fixed hyperparameters", type=str, default=None, required=False)
parser.add_argument("-gs", "--grid_search", help="whether to perform a grid search", type=bool, default=False, required=False)
parser.add_argument("-n", "--n_games", help="number of games to play per configuration", type=int, default=50, required=False)
parser.add_argument("-p", "--player", help="whether the agent should start first (0) or second (1)", type=int, default=0, 
                    choices=[0, 1], required=False)

args = parser.parse_args()

if (args.wandb is True) and (args.wandb_setup_path is None) and not (args.strategy == 'neural'):
    raise ValueError("Please provide a path to the Weights & Biases configuration file (-wp) or disable Weights & Biases (-w).")
elif (args.wandb is True) and ((args.hyp_grid is None) and (args.hyperparameters is None)) and not (args.strategy == 'neural'):
    raise ValueError("Please provide a path to the hyperparameter grid (-hg) or file (-hp) or disable Weights & Biases (-w).")


# The input is either a path (single run) or a dict (wandb)
def play_one_game(conf, input : str | dict):
    g = Game()
    strategy = conf.strategy
    # XXX: This is a bit of a mess, but it works. Revert player2 and player 1 and change player=0 and player=1.
    if conf.royal_rumble:
        player1 = StandardMCTSPlayer(g, input, player=0)
    else:
        player1 = RandomPlayer(g, player=0)
    if strategy == 'random':
        player2 = RandomPlayer(g)
    elif strategy == 'human':
        player2 = HumanPlayer()
    elif strategy == 'standard':
        player2 = StandardMCTSPlayer(g, input, player=1)
    elif strategy == 'pruned_static':
        player2 = PrunedPlayerStatic(g, input, player=1)
    elif strategy == 'pruned_dynamic':
        player2 = PrunedPlayerDynamic(g, input, player=1)
    elif strategy == 'neural':
        model = NeuralNet(g, 8, 64, 'mps')
        model.load_state_dict(torch.load(f'/Users/mattia/Documents/University/Q-uixo/games/Quixo/core/alpha_zero/checkpoints/model_8.pt'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        player2 = NeuralPlayerStandard(g, model=model, args=input, optimizer=optimizer, player=1)
    if strategy != 'neural':
        winner = g.play(player1, player2) if args.player == 1 else g.play(player2, player1)
    else:
        winner = g.play(player2, player1) if args.player == 1 else g.play(player1, player2)
    return winner, g.counter, g.get_board()

def _n_games(func, n_games=30, strategy='standard'):
    def wrapper(conf, input):
        wins = 0
        draws = 0
        moves = []
        times = []
        for _ in range(n_games):
            print(f'Game {_ + 1} / {n_games} started.')
            toc = time.time()
            winner, counter, board = func(conf, input)
            tic = time.time()
            if winner == 1:
                wins += 1
            elif winner == -1:
                draws += 1
            print(f'\n\nWinner: Player {winner if strategy != "neural" else 1-winner}')
            print(f'Number of moves: {counter}')
            print(f'Time elapsed:\n(seconds) {(tic-toc):.2f}\n(minutes) {((tic-toc)/60):.2f}\n')
            moves.append(counter)
            times.append(tic-toc)

        return wins, draws, np.array(moves), board, times
    return wrapper

n_games = _n_games(play_one_game, n_games=args.n_games, strategy=args.strategy)

if __name__ == '__main__':

    if args.wandb:
        try:
            assert args.wandb_setup_path is not None
        except AssertionError:
            raise ValueError("Please provide a path to the Weights & Biases configuration file.")
        import wandb
        with open(args.wandb_setup_path, 'r') as f:
            wandb_setup_config = yaml.load(f, Loader=yaml.SafeLoader)
        
        # Try except to handle neural config
        try:
            with open(args.hyp_grid, 'r') as f:
                hyp_grid = yaml.load(f, Loader=yaml.SafeLoader)
        except:
            pass
        
        if args.strategy == 'standard':
            grid = ParameterGrid(hyp_grid['MCTS'])
        
        elif args.strategy in ['pruned_static', 'pruned_dynamic']:
            dict_hyp = {}
            dict_hyp.update(hyp_grid['MCTS'])
            dict_hyp.update(hyp_grid['Pruner'])
            dict_hyp.update(hyp_grid['Growth_function'])
            grid = ParameterGrid(dict_hyp)
        
        else:
            dict_hyp = hyp_grid
        
        if args.grid_search:
            # grid = ParameterGrid(dict_hyp)
            
            # "This is beyond your comprehension..."
            tot_configurations = np.prod(list(map(len, (list(grid.param_grid[0].values())))))

            for run, params in enumerate(grid):
                wandb.init(
                    project=wandb_setup_config['project'],
                    name=f"{wandb_setup_config['name']}_{run}",
                    entity=wandb_setup_config['entity'],
                    config=params,
                    notes=f'Number of different configurations tried: {tot_configurations}')
                wandb.config.strategy = args.strategy

                print(f"Run {run + 1} / {tot_configurations} started.")
                print(f'Configuration: {params}')
                wins, draws, counter, _, times = n_games(conf=args, input=params)
                wandb.log({'n_wins_agent' : wins,
                        'n_draws_agent' : draws,
                        'avg_time_per_game' : np.mean(times),
                        'std_time_per_game' : np.std(times),
                        'avg_num_moves' : np.mean(counter),
                        'std_num_moves' : np.std(counter),
                        'c' : params['c'],
                        'num_searches' : params['num_searches'],
                        'beta' : params['beta_param'],
                        'temperature' : params['temperature'],
                        'starting_percentage' : params['starting_percentage'],
                        'gfunction' : params['gfunction'],
                        'amplitude' : params['amplitude'],
                        'player0' : 'StandardMCTS',
                        'player1' : args.strategy,
                        'max_num_moves' : 200,      # Change this in the __init__.py file
                        'num_searches': params['num_searches']})          
                        
                wandb.finish()
                print(f"Run {run + 1} completed.")
        
        else:
            for _ in range(10):
                wandb.init(
                    project=wandb_setup_config['project'],
                    name=wandb_setup_config['name'],
                    entity=wandb_setup_config['entity'],
                    config={},
                    notes='Neural Benchmark, Random first player.')
                wandb.config.strategy = args.strategy

                wins, draws, counter, board, times = n_games(conf=args, input=dict_hyp)
                wandb.log({'n_wins_agent' : wins,
                        'n_draws_agent' : draws,
                        'avg_time_per_game' : np.mean(times),
                        'std_time_per_game' : np.std(times),
                        'avg_num_moves' : np.mean(counter),
                        'std_num_moves' : np.std(counter),
                        'player0' : 'StandardMCTS',
                        'player1' : args.strategy,
                        'checkpoint' : 8,
                        'max_num_moves' : 200})
                wandb.finish()
    else:
        try:
            with open(args.hyperparameters, 'r') as f:
                hyp = yaml.load(f, Loader=yaml.SafeLoader)
        except:
            pass
        
        if args.strategy in ['pruned_static', 'pruned_dynamic']:
            dict_hyp = {}
            dict_hyp.update(hyp['MCTS'])
            dict_hyp.update(hyp['Pruner'])
            dict_hyp.update(hyp['Growth_function'])
        
        else:
            dict_hyp = hyp
            # dict_hyp = 'games/Quixo/config_files/standard/mcts_standard.yaml'

        wins, draws, counter, board, times = n_games(conf=args, input=dict_hyp)