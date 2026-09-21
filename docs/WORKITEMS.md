# Workitems

## Done
- FTD signal analysis report (#11): `backfire/report_signal.py` with the episode gallery and the
  scorecard, `backfire/signal_analysis.py` under it, `docs/ftd_reference.yaml` as the source of
  the reference dates, and the `event` / `day0_block` / `ftd_block` diagnostics on `FTDSignal`.

## Open
- interactive signal dashboard (#6). The chart annotations in `report_signal.panel_figure` are
  meant to be reused for it.
- `backfire/visualize_signal.py` - the README documents it but it does not exist yet.
- add `md/IXIC.csv` so the FTD signal can run on the Nasdaq Composite the reference calls were
  actually made on, rather than on QQQ as a proxy. No code change is needed, only `index: IXIC`;
  `md/` currently holds `^IXIC.csv`, which `Environment.load_ohlcv` does not find under that name.
  Some of the reference hit-or-miss noise is volume differences between the index and the ETF.
- replace the stale `notebooks/VisualizeSignal.ipynb`.
- implement vibha jha's TQQQ strategy (essentially FTD-21d) 
- trade PnL histogram into the visualization tool (tab below)
- implement claude skills encapsulating backtest
- claude skill encapsulating strategy analysis
- implement strategy analysis agent
- experiment with strategy analysis agent on 50dVS200dMACrossover strategy

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
