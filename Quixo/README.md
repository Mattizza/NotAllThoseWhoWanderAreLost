## Run the experiments!

The file `pilot.sh` orchestrates the games by letting choose the strategy (`-s`), whether the agent should play first (`-p`) and how many matches to play (`-n`). 
The different hyperparameters of the Agents (`-hp`) can be set by passing the path to a `.yaml` file, formatted as the ones in `games/Quixo/config_files/` depending on the
specific implementation. 

To test a configuration, you just need to run the following
```
$ sh pilot.sh
```

To make things easier, I have already defined some good configurations, which can be accessed by modifying `pilot.sh` as follows
```
$ python main.py -s <STRATEGY> -hp games/Quixo/config_files/<AGENT>/mcts_<AGENT>.yaml -n 10 -p 1
```

In particular, `<STRATEGY>` and `<AGENT>` must be coherent and belong to the following couplings (`STRATEGY - AGENT`):
-  `standard - standard`
-  `pruned_static - static`
-  `pruned_dynamic - dynamic`
-  `neural - neural`

You can customize the configuration you want to use by specifing your own path. However, `<STRATEGY>` must be one of those specified above.
