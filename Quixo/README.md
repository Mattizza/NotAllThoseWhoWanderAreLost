## Run the experiments!

The file `pilot.sh` orchestrates the games by letting choose the strategy (`-s`), whether the agent should play first (`-p`) and how many matches to play (`-n`). 
The different hyperparameters of the Agents (`-hp`) can be set by passing the path to a `.yaml` file, formatted as the ones in `games/Quixo/config_files/` depending on the
specific implementation. 

To test a configuration, you just need to run the following command
```
$ sh pilot.sh
```
Every Agent plays always X, thus if the winner is `Player 1` this means that the Agent has won.

---

## Configurations
To make things easier, I have already defined some good configurations, which can be accessed by modifying `pilot.sh` as follows
```
$ python main.py -s <STRATEGY> -hp games/Quixo/config_files/<AGENT>/mcts_<AGENT>.yaml -n 10 -p 1
```

In particular, `<STRATEGY>` and `<AGENT>` must be coherent and belong to the following couplings (`STRATEGY - AGENT`):
-  `standard - standard`
-  `pruned_static - static`
-  `pruned_dynamic - dynamic`
-  `neural - neural`

You can customize the configuration you want to use by specifing your own path. However, `<STRATEGY>` must be one of those specified above. Changing the player (`-p`) will result
in different startings depending on the value it receives, respectively Agent starts first if `-p 0` and Agent starts second if `-p 1`. Please note this _does not have any influence
on the fact that the Agent will always be Player 1_, playing X, but just on the order.
