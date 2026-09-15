# Workitems
- implement claude skills encapsulating backtest and analyze strategy
- trade PnL histogram into the visualization tool
- implement strategy analysis agent
- experiment with strategy analysis agent on 50dVS200dMACrossover strategy
- implement vibha jha's TQQQ strategy
- analyze vibha jha's TQQQ strategy
- implement claude skills encapsualting load_ohlcv, backtest, analyze_strategy, visualize_strategy (dashboard)
- install matt pocock skills (grill me, implement) 
- implement experiments: 
### Experiments

Every simulation run is given an experiment name which may contain a forward slash used to group experiments into groups, e.g. "index/vibha_jha_tqqq". Output is placed into directory experiments/'experiment_name', where the experiment name is first parsed and forward slashes are used as subdirectory indicators. The output directory contains information available to fully reconstruct the strategy and simulation parameters for later reruns. The output directory will contain files: 
  - f"description_{experiment_name}.md": strategy name, strategy parameters, underlying, start/end dates
  - f"stats_{experiment_name}.csv": performance metrics
  - f"trades_{experiment_name}.csv": strategy trades
  - f"pos_{experiment_name}.csv": strategy positions
  - f"entry_{experiment_name}.csv": entry signal daily data
  - f"exit_{experiment_name}.csv": exit signal daily data


- implement and analyze: 
  - canslim (roppel): MarketState, CupAndHandle, PullbackBreakout, FlatBaseBreakout, Breakdown indicators
  - minervini (vcp): Stage2, VCP, violation exit strategies

- implement research agent
- implement auto research agent



# Issues and Defects
