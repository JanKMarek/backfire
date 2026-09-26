# Backfire - quantitative stock investing strategy research engine. 

## Project Goals and Overview
Backfire is a quantitative stock investing strategy research engine aimed at quantitative and qualitative strategy behaviour understanding, optimization and development. 

Purpose: 
- verify claims made by popularizers of individual retail investor strategies (e.g., Vibha Jha, Mark Minervini, Jim Roppel, IBD) by backtesting the strategies on historical data and analyzing their performance and risk/reward profiles 
- explore parameter value variations (e.g., different moving average periods) and behaviour variations (e.g., different exit criteria) and their impact on strategy reward and risk
- find optimal parameter sets for strategies given an investor risk/reward profile
- auto-research modifications to strategies to improve their suitability given a target (risk/reward profile), constraints (holding period, etc)

Strategy behaviour is analyzed by executing deterministic simulations of strategies against historical and synthetic data and analyzing the generated quantitative metrics, trades and signals. 

Strategy result analysis and optimization is performed by LLM-based agents following instructions and double checked by visual inspection of the results. 

Auto research of strategy modifications will be performed by LLM-based agents to leverage their planning/reasoning and deep research capabilities of frontier LLM models. 

The goal of the research engine is answering questions such as: 
- 'Make strategy using the FTD signal as entry point and TakeProfit at 25% as exit.': composes a strategy from existing library of signals. 
- 'Run strategy ftd_takeprofit': simulate strategy with given parameters and determine the risk, return and individual trades the strategy would  generate over a trading period (quantitative metrics).
- 'Analyze strategy ftd_takeprofit': understand the strategy behaviour: when/how did the strategy make money/lose money, did it have few/many winning trades vs. many/few losing trades, did it have one big win/loss driving the results, what was the average trade holding periods, how did it behave around market turning points, how long was it in drawdown, etc. 
- 'Identify weaknesses of straetgy ftd_takeprofit': what are the weaknesses of a strategy, e.g. dependency on one big winning trade, sensitivity to slippage (trading at opening price), what risks are not mitigated in this strategy? 
- 'Run ftd_takeprofit with several takeprofit parameters': would changing strategy parameters or rules result in a better or worse outcome? 
- 'Optimize ftd_takeprofit. Use TQQQ 2000-20019 as training set, 2020-2016 as test set': what is the optimal set of parameters for a strategy? 
- 'The target is 30% CAGR, constraints trading TQQQ on daily data with no more than 30% unrealized maximum drawdown. Suggest a strategy that fits these criteria.': how could this strategy be modified (parameters, rules of behaviour) to improve its performance?
- '': how would a strategy behave on a synthetic/simulated market data scenario (terrorist attack, etc.)

The research engine supports: 
- index, ETF and individual stock strategies operating on daily OHLC data
- strategies composed of entry signals, exit signals, risk management rules and position size rules
- individual signal visualization and metrics
- strategy performance simulation/backtest, behaviour visualization and qualitative performance metrics analysis
- manual and automated strategy analysis, optimization and improvement  

The engine does not support the following yet: 
- shorting instruments
- intraday trading
- options trading
- trading a portfolio of instruments
- cryptocurrency trading

## Application Domain 

### Domain
The research engine analyzes long-only trading strategies operating on stocks, stocks indexes and ETFs. The strategy decides its behaviour and executes trades based on market data. Market data is provided in the OHLCV format - open, high, low, close and volume. 

### What is a strategy
We define a stock investing strategy as a set of rules for buying and selling an underlying stock market instrument(s).

The strategy logic simulates actions of a trader who: 
- checks the state of the market at the end of the day (and so has access to open, close, high and low prices of the underlyings for the current day)
- decides on actions (buy to open a positon, add to an existing position, sell to close a position, sell to reduce a postiion, do nothing)
- executes the actions at the beginning of the next day (executing at the Open price of the underlying the next day).

Note that buying/selling at the next day open may not not be possible which is a source of slippage in backtesting the strategy. 

Examples of underlying stock market instruments: 
- index/sector ETFs (e.g., SPY, TQQQ, SOXL)
- individual stocks (e.g., MU, ANET)

All prices are denominated in US Dollars (USD). 

Strategy rules: 
- **general overlay rules**: decide when the strategy is inactive (a.k.a 'in cash', 'out of the market') or active (i.e., has positions in the market). General overlay rules may be market related (market in downtrend, market choppy). General overlay rules take precedence over buy/sell rules. 
- **selection rules**: what underlying should the strategy buy/sell. Examples: trade TQQQ only, select top ticker based on relative strength and fundamental growth from a sector. 
- **buy rules**: triggers that cause the strategy to buy the underlying. Examples: entry buy rule to establish the initial position in an underlying, add-on rule to add to a winning position (may be added in the future)
- **sell rules**: triggers causing the strategy to sell the underlying. Examples of these are selling on predefined profit target (selling into strength while the underlying is still advancing), selling on a retracement (selling into weakness after the underlying has begun to decline) or sell rules based on other indicators. 
- **risk management rules**: triggers protecting portfolio capital against. Examples include initial stop loss set a predefined amount below the buy point or cool-off period enforced after a period of losing trades. Risk management rules take precendence over buy/sell rules and general overlay rules.  
- **position size rules**: determine how much of the underlying (e.g., the number of shares or total dollar position size) should the strategy buy or sell.  

Here is an example of a strategy: 
Strategy "Index 50d/200d MA Crossover". 
- general overlay: none
- selection rules: always buy/sell QQQ
- buy rules: when 50d moving average (MA) crosses above 200d MA from below, buy QQQ
- sell rules: when 50d MA drops below 200d MA, sell QQQ
- position size rules: always buy/sell as many shares as you can (so spend all portfolio cash) 

And yet another example: 
Strategy "Vibha Jha TQQQ index strategy": 
- general overlay: none
- selection rules: always buy/sell TQQQ
- buy rules: on Follow-Through Day or three Higher Highs/Higher Lows after QQQ consolidation of 8%-10% over at least four weeks
- sell rules: 
  - strength take profit rule: trade profit 25%+ AND QQQ 10% over the 21d MA
  - weakness take profit rule: QQQ closes below 21d MA for two days in a row
- risk management rules: 
  - initial stop loss: QQQ closes below the First Day of Rally low of the day
- position size: 
  - buy TQQQ shares for 50% of the portfolio equity

Another example of a strategy: 
Strategy "CANSLIM Market Leader Breakout, Single Run": 
- general overlay: general market in uptrend 
- selection rules: select most promising market leader breaking out from a consolidation
- buy rules: move above the pivot point on volume
- sell rules: 
  - take profit rule (aka sell into strength): trade profit 25%+ and stock price 10%+ over the 21d MA
  - retracement sell rule (aka sell on weakness): stock price closes below 21d MA
- risk management rules: 
  - initial stop loss: 7% below pivot point
- position size rules: buy shares for 20% of the portfolio equity

  ### Strategy signals and rules

To simplify reasoning about strategies and strategy analysis, it is useful to construct a strategy from signals.  

Signals generate values (binary or continuous) at the end of every day. For binary signals, whenever the signal value goes from False to True (we say it triggers or it fires), we generate a signal id for that 'firing' and the same signal id is used as long as the signals stays on. When a new signal is re-generated (i.e., signal values goes to False and then to True again), the signal id is incremented. Some signals trigger only for one day (e.g., price crosses a MA to the upside), others stay on (price above a MA). Examples of signals: ShortMAAboveLongMA, TakeProfitSignal. 

A signal-driven strategy is then composed of: 
- **general market overlay signal**: signal usually calculated from general market averages (e.g., SPY or QQQ indexes). It may be a binary signal (strategy is free to buy when the signal is on) or a continuous signal (which may indicate the level of exposure the strategy should be taking on). Strategies operating on indexes generally do not need a separate market overlay signal since the general market information will be reflected in the index signals and so will generally need no market overlay signal. 
- **entry signal**: When this signal triggers, the strategy establishes position with size determined by the the position management rules. The same signal will not be reentered - i.e., if a position was exited (based on SL, TP, TS or ExitSignal), the position will not be reentered until an entry signal with a different signal id is triggered. A new signal will also not be entered as long as the exit signal which caused it to be exited is in place. Note that entry signal going off (i.e., from True to False) does not mean exiting the position - only exit signal triggers position exit. Use negation of entry signal as exit signal if you need such behaviour. 
- **exit signal**: When this signal triggers, it takes precedence over the entry signal and full position is exited. When it goes off, the position may not be reentered on the same entry signal (strategy needs a new entry signal to put on a position) and may not be reentered as long as an exit signal is on. Common exit signals: 
  - take profit (TP): when in position and trade gain exceeds a threshold, close the position and do not act until a new entry signal is triggered
  - trailing stop (TS): when price falls back below a threshold (e.g., 21d MA or -10%), close the position and do not act until a new entry signal is triggered
  - exit signals calculated from the values of the underlying, e.g., price falls below a moving average.
- **risk management rules**: 
  - initial stop loss rule (SL): when in position and trade loss exceeds a threshold, close the position and do not act until a new entry signal is triggered
  - cool-off period (CP): after a sequence of losing trades
  **position size management rules**: 
    - Fixed Fraction: invest a percentage of the portfolio equity on entry, sell the entire positon on exit.  
    - Gradual Exposure (may be implemented in the future): as long as the same entry signal remains in place, the position size is increased as the underlying appreciates based on a predefined schedule, e.g. 10% initially, add 5% on 5% up and add 5% on 10% up. The position will NOT be decreased as the underlying drops (exit is triggered by SL, TP, TS or exit signal). 

Examples of strategies in yaml format are in directory strategies. 

### Strategy simulation 



Strategy simulation takes the following parameters: 
  - start date, end date
  - underlying (ticker)
  - strategy definition (yaml file with command line overrides)
  - market data location
  - out : persistent output folder ("" means no persistent output)

In the simulation, the strategy starts with a certain portfolio size in cash, executes its rules over the trading period and closes all positions at the end of the trading period.  

The simulation framework generates performance statistics, list of trades, strategy position daily values and signal daily values: 

Performance statistics/metrics: 
- number of trades: total (N_total), winning (N_win), losing (N_loss).
- winning trade percentage: N_win / N_total.
- average winning trade gain: W_avg (e.g., 0.24).
- average losing trade loss: L_avg (e.g., -0.33).
- R: W_avg / abs(L_avg), reported as NA if L_avg is zero.
- max winning trade return: max_pnl_pcnt (e.g., 1.34).
- max losing trade loss: min_pnl_pcnt (e.g., -0.98) 
- holding period: across all trades (hp_avg), over winning trades (hp_win), over losing trades (hp_loss). In business days.
- absolute dollar profit/loss: over all trades (total_pnl), over winning trades (positive_pnl), over losing trades (negative_pnl)
- return: (portfolio size after last trading day / initial portfolio size) - 1
- compounded annualized growth rate: CAGR calculated as power((portfolio_end/portfolio_start), 1/holding_period_in_years) - 1
- maximum realized drawdown: max_dd_pcnt_realized
- maximum unrealized drawdown: max_dd_pcnt_unrealized
- maximum time in drawdown: max_time_in_dd, the longest stretch in calendar days that the daily mark to market balance spends below a previous high water mark - measured from the day the mark was set to the last day before the balance recovers it, or to the last trading day if it never does.
- Sharpe ratio: sharpe, the mean of the daily mark to market returns over their standard deviation, annualized by sqrt(252). Measured against a zero risk free rate, so it is comparable across runs but not against a published Sharpe. Reported as NA if the balance never moves.
- Calmar ratio: calmar, CAGR / max_dd_pcnt_unrealized. Reported as NA if the strategy never drew down.

List of trades executed by the strategy. For each trade: 
- trade_number: trade number
- ticker: underlying
- entry_date
- entry_price
- shares: number of shares bought or sold
- exit_date
- exit_price
- memo:  string. For instance, Memo on entry "bought:xxx shares:entry_signal_name", Memo on exit "sold:xxx shares:SL/TP/TS/exit_signal_name"
- pnl: dollar return on the trade
- pnl_pcnt: return on the trade calculated as (exit_price / entry_price) - 1
- hp: holding period in business days

Strategy position daily values - for every day in the trading period: 
- Date
- O,H,L,C,V prices
- es,es_id,xs: entry signal, entry signal id, exit signal
- pos: position at the end of the day (number of shares of the underlying) 
- cash: cash at the end of the day (in USD)
- action: action to execute the next day at the market open
- delta_shares: the number of shares to buy/sell the next day at the market open
- memo: reason for the action (signal name)
- buy_price: price at which shares were bought at the market open of the current day
- balance: 
- unrealized_CumMax
- unrealized_dd,
- unrealized_dd_pcnt

Signal daily values - for each signal a file with one row for every day in the trading period: 
- Date
- signal value 
- signal id


All dates in YYYY-MM-DD. All stock prices reported with two decimal places. All Pnl/gain/loss numbers with zero decimal places and comma at thousands. All percentages with two decimal places, e.g. 0.77 or -0.33. 


### Experiments

An experiment is created whenever a simulation is needed to confirm or disprove a hypothesis. Experiments can contain one simulation run (e.g., analyzing a strategy with fixed set of parameters), several simulation runs (e.g, optimizing parameter sets given constraints) or a set of subexperiments (e.g., spawning several subagents to explore several research directions in autoresearch). Experiments are generated by tools like strategy simulator, analyzer, optimizer and researcher. 

An experiment persists simulation results as well as all information necessary to reproduce the experiment later - descriptive experiment name, experiment definition, simulation runs performed within the experiment, experiment conclusions. For each simulation run, it stores strategy parameters, the underlying, start/end dates and results.

An experiment includes: 
- descriptive experiment name: either provided or generated from the experiment intent
- experiment definition: 
  - goals:  e.g., return exceeds market index CAGR over the same time period
  - constraints: on parameters, rules or outcome (e.g., realized drawdown no more than 20%)
  - validation: training (in-sample), validation and holdout (test) data
  - resources: maximum number of iterations, length of time, models used, etc. 
  - further instructions: e.g., optimize for plateaus (report median metric over parameter neighbourhood, heatmaps)
- strategy definition and strategy parameters
- experiment results: 
  - number of trials
  - metrics generated by trials - median values, confidence intervals, etc reported for training, validation and test datasets. 
- experiment conclusion: 
  - conclusion based on analysis of experiment outcomes
  - comparison to benchmark (e.g., buy-and-hold strategy)
  
Examples of experiments: 

1./ Did Vibha Jha's TQQQ strategy with default parameters on TQQQ return more than 50% CAGR from 2020 to 2026? 
- name: "Vibha Jha TQQQ 2020-2026 backtest"
- definition: underlying TQQQ, period 2020-2026, goal and constraints are not applicable
- strategy: Vibha Jha TQQQ (in strategies/*) with standard FTD parameters and 21d MA for exit
In this experiment, we check the claim Vibha Jha is making and also understand the behaviour of the strateegy. The agent would construct a definition of the strategy, download the market data, invoke the backtest tool, save the results and report the results.

2./ Understand the behaviour of Vibha Jha's TQQQ strategy - what are its strengths, weaknesses, what risks are not mitigated by the strategy? 
The analysis agent would use the results saved from the previous run to analyze the results using the strategy_analysis skill. Additionally, it would suggest the command to launch the visualization tool to further understand the strategy. 

3./ Vary parameters of Vibha Jha's TQQQ strategy over defined intervals and find the parameter set with the highest Calmar ratio.
Also vary SL, TP and TL parameters (including not using them). Use brute force for parameter space search. 
- name:  "Vibha Jha TQQQ 2020-2026 best Calmar ratio"
- definition: goal: maximum Calmar ratio (return to realized drawdown), constraints: exit MA period one of 10, 21, 50, 100 days.
- strategy: Vibha Jha TQQQ (in strategies/*), standard FTD parameters, MA for exit varies as do SL, TP and TL parameters.  
In this case, we search the parameter space for the best parameter set. The optimization agent would perform a set of simulation runs using the backtest local tool to search the parameter space by brute force (as instructed) and would select the parameter set with the highest Calmar ratio. The results of the search would be saved for later inspection with visualization tools. 

3./ Improve the Vibha Jha's TQQQ strategy by modifying its parameters and rules as necessary within constraints. 
- name: "Vibha Jha TQQQ strategy autoresearch"
- experiment definition: underlying TQQQ, goal CAGR 20%+, constraints maxDD better than 20%, average holding period 2wk+, vary entry signals, exit signals
- strategy: Vibha Jha TQQQ as the starting point 
In this case, the autoresearch agent would iterate over strategy modifications until the target is reached. It might launch several optimization agents in parallel (each investigating a different branch) and use the analysis agent to determine the directions of research. 

### Strategy evaluation
Strategy evaluation is the process of comparing strategy performance to its objectives and constraints and analyzing its behaviour: 

Strategy Objectives: 
  - total return over trading period
  - CAGR (compounded annualized growth rate)
  - Calmar ratio (CAGR / MaxDrawdown)
  - Sharpe ratio

Constraints: any of the following 
- maximum drawdown (typically unrealized) 
- time in drawdown (in trading days)

Trades placed by the strategy: 
- number of trades
- expected trade return, std deviation of returns, min/max return
- average trade holding period


Red flags: 
- many small losing trades offset by very few big winning trades (dependency on single/few big wins) 
- very large maximum losing trade
- too many trades, holding period too short
- sensitivity to slippage (e.g., execution on opening price vs. execution on price on open+15min) 
- sensitivity to small parameter changes (overfitting) 
- presence of unmitigated risks
- long periods of time spent in drawdown (emotional impact)

Positive traits: 
- can be leveraged

Visual Inspection of strategy performance is also helpful. We chart the underlying over the trading period overlaid with: 
- strategy actions (buys/sells) 
- strategy unrealized PnL
- signal values (entry, exit, general market)

Strategy result visualization tool uses data generated by the backtest/simulation framework and displays an interactive dashboard containing: 
- Header: strategy name and backtest period; key statistics (total return, CAGR, three largest drawdowns) 
- Chart Pane: timeseries chart showing underlying, entry and exit signals and the strategy unrealized PnL (portfolio value). It is possible to zoome in on a part of the chart. 
- Data Pane: this is a tabbed pane with two tabs - Monthly Returns and Trades. 
  - The Monthly Returns tab holds a display of monthly strategy returns and annual returns (along the bottom line). Clicking on a cell in thsi table highlights the beginning and end of the corresponding time period in the Chart Pane. 
  - The Trades tab lists trades in a sortable table. The table contains trade entry date and price, exit date, price and reason; and trade pnl. The table is initially sorted by entry dates. The exit reason is the exit signal's name, 'stop loss' or 'end of backtest'; the table also shows the trade number, return and days held. When a trade row is selected, the trade's entry and exit points are highlighted in the Chart Pane. 

### Strategy analysis
To be done. 

### Strategy optimization 
To be done. 

### Strategy research
To be done. 






## Architecture and Design

### Market Data
Market data is cached in folder md and downloaded as needed using tool tools/download_md.py. Each ticker is stored in a csv file containing OHLCV elements and loaded into a dataframe with columns OHLCV. 

### Experiment

An experiment is created whenever a simulation is needed to confirm or disprove a hypothesis. Experiments can contain one simulation run (e.g., analyzing a strategy with fixed set of parameters), several simulation runs (e.g, optimizing parameter sets given constraints) or a set of subexperiments (e.g., spawning several subagents to explore several research directions in autoresearch). 

An experiment persists all information necessary to rerun the simulation later - descriptive experiment name, experiment definition, simulation runs performed within the experiment, experiment conclusions. For each simulation run, it stores strategy parameters, the underlying, start/end dates and results.  

System components (simulator, visualizer, optimizers) communicate by using persisted experiment results. 

### Strategies and signals 
Signal definition templates are in strategies/signals stored as yaml files. 

Strategy definition templates are in strategies. 

Overrides to signal/strategy templates are provided via command line parameters provided to the simulator. 

### Evaluator 
Computes strategy run metrics. 

### Simulator 
Script backtest.py encapsulates the entire process of a single backtest run - load market data, load strategy, evaluate signals and generate trades, evaluate statistics, save evaluation data to experiment output files and return evaluation data to the caller. 

### Visualizers
Visualizers use simulation results to visualize results in static HTML pages, interactive dashboards or interactive jupyter notebooks. 

### Analyzers 
AI agents use skills to execute strategy analysis. 

### Optimizer
To be done. 

### Researcher
To be done. 


## CLI Workflows

### Project Setup
```
git clone https://github.com/JanKMarek/backfire.git
uv sync                                    # create .venv and install everything (incl. dev group)
```
Run all commands from the repository root. 


### Strategy simulation 

Create a yaml strategy definition (see strategies directory for strategy definition templates): 

```yaml
strategy: 
  name: ShortMAVsLongMA
  entry_signal: 
    name: ShortMAAboveLongMA
    short_MA: 50 # days
    long_MA: 200 # days
  exit_signal: 
    name: ShortMABelowLongMA
    short_MA: 50 # days
    long_MA: 200 # days
  risk_management: 
    name: BasicRiskManagement
    stop_loss: 0.07
  position_management: 
    name: PositionManagement
    initial_position: 100000
    policy: fixed_fraction
    fraction: 1.0      
```

Run the simulation (backtest): 

`uv run python backfire/backtest.py --start_date "2020-01-01" --end_date "2026-07-01" -md "md" --underlying QQQ --strategy strategies/ma_crossover.yaml --out "out/my_strategy"`

The strategy file names the component classes and their constructor arguments. Any Signal,
BasicRiskManagement or PositionManagement subclass defined in `backfire.base` or `backfire.signals`
can be used by its class name; `exit_signal`, `risk_management` and `position_management` are
optional and fall back to the strategy defaults. An argument that is itself a mapping with a
`name` is built as a nested signal, which is how combinators are configured:

```yaml
  exit_signal: 
    name: ReverseSignal
    signal: 
      name: ShortMAAboveLongMA
      short_MA: 50
      long_MA: 200
```

A combinator taking any number of signals, such as `OrSignal` (on whenever at least one of its
signals is on), is given a list of them. Rules on the open trade are exit signals too:
`TakeProfitSignal` is on once the trade has gained `threshold` over its buy price (selling into
strength), and `TrailingTakeProfitSignal` is armed once the trade has gained `threshold` and is then on
whenever the price closes below the `period` day moving average (selling into weakness).
`RetracementSignal` is a trailing stop: it is on once the price closes more than `retracement` below
the highest high since the position was opened. Selling on a market signal, into strength or into
weakness is written as:

```yaml
  exit_signal:
    name: OrSignal
    signals:
      - name: ShortMABelowLongMA
        short_MA: 50
        long_MA: 200
      - name: TakeProfitSignal
        threshold: 0.25
      - name: TrailingTakeProfitSignal
        threshold: 0.2
        period: 10
      - name: RetracementSignal
        retracement: 0.1
```

`BasicRiskManagement` keeps the rule that protects capital, `stop_loss`: the initial stop, a
fraction below the buy price.

To list all signals that can be named in the strategy yaml file, run: 
`uv run python backfire/backtest.py --list_signals`

Individual entries can be overridden per run without editing the strategy file - the path is
dotted from the root of the file and the value is read as YAML, so types are preserved:

`uv run python backfire/backtest.py ... --set strategy.entry_signal.short_MA=20 --set strategy.risk_management.stop_loss=null`

`--out ""` (the default) runs the backtest without writing any persistent output.

### Visualizing strategy results

Command `uv run python backfire/visualize.py -out "out/test/Index_50dMAvs200dMA"` launches the web server; point browser at http://127.0.0.1:8050/ to view the dashboard. The dashboard reads the CSV files the backtest wrote into the `--out` folder; it does not re-run the backtest. `--host` and `--port` override the default `127.0.0.1:8050`, and `--debug` runs the Dash development server with the reloader and the in-browser error pane.

### Visualizing an individual signal
Visualize signal using visualize_signal.py. This module takes signal definition from a yaml file. If no signal definition file is provided, it collects signal definition from all command line arguments starting with 'signal.'. If a signal definition file is provided, any command line arguments starting with 'signal.' override the values in the signal definition file. 

The module computes signal values using logic consistent with backtest.py. It generates signal value output files it will need for visualization into a temporary directory 'temp'. 


Yaml file for the signal (see `docs/SIGNALS.md` for what the FTD parameters mean): 
```yaml
  signal:
    name: FTDSignal
    index: ^IXIC
    min_decline: 0.08
    min_peak_age_days: 20
    day0_window: 5
    ftd_min_gain: 0.0125
    ftd_min_days: 4
    ftd_max_days: 25
```


Run 
`uv run python backfire/visualize_signal.py --signal my_signal.yaml --signal.name=FTDSignal --signal.ftd_min_gain=0.015 --signal.day0_window=10`

### FTD signal verification report

`backfire/ftd_signal_verification_report.py` writes the verification report that `docs/SIGNALS.md`, "Signal
verification", specifies: one signal over one underlying, scored against a set of ground truth
dates, as a static HTML file.

```
uv run python backfire/ftd_signal_verification_report.py --underlying QQQ --start_date 1999-03-10 \
    --signal strategies/signals/ftd.yaml --ground_truth turnarounds --out out/ftd_verification
```

It takes the same `--signal` file plus `--signal.*` override convention as `visualize_signal.py`,
and the same signal catalog as `backtest.py`. With no `--signal` file the overrides define the
signal on their own.

`--ground_truth` picks what the signal is verified against:

- `turnarounds` (the default) - `docs/turnaround_points.csv`, the market turnarounds picked
  visually with hindsight: does the signal meet its intent?
- `ibd` - the `reference` entries of `docs/ftd_reference.yaml`, IBD's own follow-through day
  calls: does the signal fire when IBD did? The file's `candidate` entries are recalled but
  unconfirmed dates and are not scored.
- a path to a `.csv` or `.yaml` file in either format.

A firing is a *positive*. It is a *true positive* when it falls within the tolerance of a ground
truth date - at most `--early_days` trading days before it (default 1) or `--late_days` after it
(default 3) - and each firing can match only one date; every other firing is a *false positive*.
Precision is true positives over positives, recall is true positives over the ground truth dates
the data covers, and F1 their harmonic mean.

The run writes four files into `--out`:

- `ftd_signal_verification_report.html` - self-contained, with plotly.js inlined, so it opens
  offline
- `scorecard.csv` - one row per ground truth date: the firing that matched it and its gap, or the
  reason nothing did, plus the state and the block marks the signal recorded around the date
- `episodes.csv` - one row per rally attempt: its Day 0, Day 1, peak, decline, outcome, and for
  a confirmed one the follow-through day, its gain and volume ratio, and whether it later failed
- the signal's daily values, the same file a backtest saves

The page has these parts:

- a **header** with the statistics: ground truth dates in the data, positives, trading days,
  true and false positives, precision, recall and F1
- the **parameters** of the signal run (class, constructor arguments, underlying, period) and of
  the verification test (ground truth file, tolerance, forward horizon), so a run can be repeated
- a **scorecard**: the ground truth dates and the false positives in one table, in date order. A
  ground truth date shows the firing date, the reason for not firing and the note recorded for
  the date; a false positive its firing date. Every firing also shows its rally day, gain and what
  became of the uptrend, and each row links to its chart
- **what followed a firing**: the average path of the underlying for up to 60 trading days
  for all follow-through days (with an interquartile band), the successful ones (the uptrend
  held until a new correction) and the failed ones, a table of their median return at 5, 10,
  20, 40 and 60 days, and the forward returns at 5, 20 and 60 days against two
  baselines - every day, and "naive follow-through days" that apply the gain and volume tests
  with none of the correction, Day 0 or day-count logic around them. Measured from the **next
  day's open** the way `SignalDrivenStrategy` executes a signal
- **the rally attempts**: how they ended, the rally day the follow-through day lands on, and a
  collapsed table of the attempts that were undercut or timed out
- the **sensitivity** of the statistics to the parameters, one parameter away from the run's
  setting at a time and scored against the same ground truth
- a **gallery**: one annotated chart per ground truth date, in scorecard order, then the false
  positives folded away. Each shows the candles and volume, the peak and the decline to Day 0, the
  rally low, the rally-day numbers, the follow-through day, × marks on the days that were blocked
  from being a Day 0 or a follow-through day with the reason on hover, a dashed line on the ground
  truth date, and a band of the state machine's state. Hovering a bar shows its whole diagnostic
  row.

The reason for a miss is read off the signal's own diagnostics: a firing close by but outside the
tolerance, the uptrend the signal was still in, the clause that blocked the Day 0 (`DECLINE`,
`PEAK_AGE`) or the clause that blocked the follow-through (`TOO_EARLY`, `VOLUME`).

The report reads the day-by-day diagnostics `FTDSignal` records (see `docs/SIGNALS.md`), so it
only runs for that signal; another signal is refused with a message saying so. About a hundred
charts is a heavy page, so a chart is drawn only when it scrolls into view.


### Running visualization notebooks: 
`uv run jupyter lab`
Note these are not updated yet. 

### Run fast unit tests
`uv run pytest test/unit -q`

### Run full test suite
`uv run pytest test -q`







