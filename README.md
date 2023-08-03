#MUFO: Multi-UAV Flight Optimization for Enhancing Connectivity in Remote Driving Services



## Requirements

```
python==3.7 
numpy==1.18.5 
keras==2.4.3 
tensorflow==2.3.0 
matplotlib==3.3.0
scikit-image==0.16.2 
tqdm==4.45.0
```

## Signal map information is formatted as PNG file one pixel representing on signal cells. 

* red vulnerable cell 
* magenta no-fly cell
* orange heavy traffic areas
* blue	driving road (EV moves)
* yellow ABS depot

The tester can create a new map by any tool which can generate PNG with above color codes.

The terrestrial map to know line-of-sight (LoS) or non-line-of-sight (NLoS) info can be based on a pre-defined map as in remotedriving.npy (Shadowing file)



## How to use

Train a new model with your map (in the image folder).

```
python train.py --gpu --config config/remotedriving.json --id signal-map

--gpu                       Activates GPU acceleration for training
--config                    Path to config file in json format
--id                        Overrides standard name for logfiles and model
```

```
python evrelay.py --weights models/trained-model --config config/remotedriving.json --id signal-map --samples 30


--Test                      Path to weights of trained model
--weights                   Path to weights of trained model
--config                    Path to config file in json format (hyperparameters)
--id                        Name for exported files
--samples                   Number of Monte Carlo over random scenario parameters
--seed                      Seed for repeatability
--visual                    Plot data
--num_agents                Overrides number of agents range, e.g. 12 for random range of [1,2] agents, or 11 for single agent
```

