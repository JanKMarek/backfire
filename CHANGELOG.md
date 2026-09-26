# September 24, 2026 - Trades tab in the strategy dashboard

The Data Pane of `backfire/visualize.py` is now tabbed: Monthly Returns (as before) and Trades.

- **Trades tab.** A sortable table of the run's trades, initially by entry date: entry date and
  price, exit date and price, exit reason, P&L, return and days held. The exit reason is read from
  the trade memo: the exit signal's name, 'stop loss' or 'end of backtest'. Return is computed
  from the prices, as the stored `pnl_pcnt` is rounded to 0.01.
- **Selecting a trade** draws a translucent band from its entry to its exit in the Chart Pane, with
  rings on the two markers; the zoom is kept. Each tab keeps its own selection and the chart shows
  the one of the visible tab. Sorting the table clears the selection.
- **Fix.** The clicked cell of the Monthly Returns table is now yellow; Dash's own pink active
  cell colour used to hide it.

# September 23, 2026 - FTD signal verification report

`backfire/ftd_signal_verification_report.py`, renamed from `backfire/report_signal.py`, now
writes the verification report that `docs/SIGNALS.md`, 'Signal verification', specifies, in place
of the analysis report of September 21.

```
uv run python backfire/ftd_signal_verification_report.py --underlying QQQ --start_date 1999-03-10 \
    --signal strategies/signals/ftd.yaml --ground_truth turnarounds --out out/ftd_verification
```

- **Ground truth.** `--ground_truth turnarounds` scores the signal against the turnaround points
  of `docs/turnaround_points.csv` (does the signal meet its intent?), `--ground_truth ibd`
  against the reference calls of `docs/ftd_reference.yaml` (does it fire when IBD did?); a file
  path in either format also works. The candidates of the YAML are not scored. `--references` is
  gone.
- **Matching.** A firing matches a ground truth date when it falls from `--early_days` (1)
  trading days before it to `--late_days` (3) after it, replacing the symmetric 10 day
  tolerance, and each firing matches at most one date. A firing is a positive, a matched one a
  true positive, the rest false positives; the report gives precision, recall and F1.
- **Contents.** The statistics, the parameters of the run and of the test, a scorecard with one
  row per ground truth date - the firing date, the reason for not firing, the note - and one per
  false positive, all in date order in a single table, then a gallery of one chart per ground truth date and per false positive
  linked from the scorecard. The forward returns against the baselines, the rally attempt
  summary, the rally day histogram, the unconfirmed attempts and the parameter sensitivity table
  stay, the last now scored against the chosen ground truth with the same tolerance.
- **Files.** `ftd_signal_verification_report.html`, `scorecard.csv` (was `references.csv`),
  `episodes.csv` and the signal values.
- **`backfire/signal_analysis.py`** gains `load_turnaround_points`, `load_ground_truth`,
  `match_ground_truth` (replacing `match_references`), `false_positive_dates` and
  `verification_stats`; `sensitivity` scores against a ground truth list with the same
  tolerance and reports `hits`, `ground_truth`, `precision` and `recall`.
- **Successful against failed.** The average path chart draws one line each for all, the
  successful (ended in a new correction) and the failed follow-through days, and a new table gives
  their median return at 5, 10, 20, 40 and 60 days. `signal_analysis.py` gains
  `ftd_dates_by_fate` and `path_table`; `run_report` returns the table as `path_stats`.

# September 21, 2026 - signal analysis report

Added `backfire/report_signal.py`, a CLI that writes a static HTML report about one signal over
one underlying, and the two things it is built on.

```
uv run python backfire/report_signal.py --underlying QQQ --start_date 1999-03-10 \
    --signal strategies/signals/ftd.yaml --out out/ftd_report
```

The report answers two questions separately. *Does every firing follow the rules?* - a gallery of
one annotated chart per episode, the reference dates first (matched, then missed), then the
candidates, then the firings no reference accounts for; each shows the peak and the decline to
Day 0, the rally low, the rally day numbers, the follow-through day, × marks on the days blocked
from being a Day 0 or a follow-through day with the reason on hover, and a band of the state
machine's state. *Does a firing mark a real turnaround?* - a scorecard of forward returns against
two baselines (every day, and "naive follow-through days" with none of the correction or day
count logic), the average forward path with an interquartile band, and a parameter sensitivity
table. Forward returns are measured from the next day's open, the way `SignalDrivenStrategy`
executes a signal. Next to `report.html` the run writes `episodes.csv`, `references.csv` and the
signal's daily values. See the README for the full description.

- **`FTDSignal` now explains itself.** Three columns join the daily values: `event` (the
  transitions the day made - `DAY0`, `DAY0_LOWERED`, `DAY1`, `UNDERCUT`, `TIMEOUT`, `FTD`,
  `FTD_FAILED`, `NEW_CORRECTION`, joined with `+` when a day makes more than one), `day0_block`
  (`DECLINE` / `PEAK_AGE`, on a new low that was not a Day 0) and `ftd_block` (`TOO_EARLY` /
  `VOLUME`, on a rally day that gained enough but was not a follow-through day). `es` is on
  exactly where `event` contains `FTD`. The rules themselves did not change: the same 62 firings
  on QQQ. Like `es`, the three mark a single day - never carried over, and moved to the next
  trading day when the underlying did not trade.
- **The reference dates are data.** `docs/ftd_reference.yaml` holds the 12 reference and 29
  candidate follow-through days that `docs/SIGNALS.md` tabulates, and is now the copy the code
  reads - the integration test and the report both take it from there. The tables in SIGNALS.md
  stay for reading.
- **`backfire/signal_analysis.py`** is the pure pandas layer under the report, usable from a
  notebook: `extract_episodes`, `forward_outcomes`, `forward_paths`, `baseline_dates`,
  `match_references` and `sensitivity`. Only `extract_episodes` is specific to the follow-through
  day; the rest work off any signal's rising edges.

One documentation error came out of this. `docs/SIGNALS.md` recorded the Sep 2010 reference as
out of reach because the late-August decline never reached 8% below the Aug 9 high. It does reach
8.9%; what rejects it is the age of that high - 14 trading days before the Aug 27 low, short of
the 20 required. The recorded reason is `PEAK_AGE`, as it is for 2009 and 2022, and SIGNALS.md
has been corrected. The integration test now asserts the recorded reason for all four
unreachable references rather than only that they are unreachable.

# September 21, 2026

Reimplemented `FTDSignal` against the specification in `docs/SIGNALS.md` as an explicit four
state machine - `WATCHING`, `DAY0`, `RALLY`, `UPTREND` - whose transitions follow the spec rule
by rule. What changed for anyone using it:

- The signal is now on for **one day only**, the Follow Through Day itself, at most once per
  rally attempt. It used to stay on for a whole uptrend and go off on a retracement from the
  highest close. A strategy therefore enters on the Follow Through Day and relies entirely on
  its exit signal to leave; after an exit it waits for the next Follow Through Day, which on
  QQQ can be months or more than a year away. Expect materially fewer trades.
- A rally attempt is now only looked for while the market is in a correction: the day 0 low has
  to be at least `min_decline` below a peak at least `min_peak_age_days` old. Day 0 is an
  explicit state that slides down to lower lows until day 1, the first day after it that closes
  up. After a Follow Through Day the signal re-arms either when the index closes below the
  confirmed rally's low (a failed Follow Through Day, which keeps the original correction's
  peak) or when it declines far enough from the high made since.
- Parameters: `lookback_days` is now `day0_window` and `terminal_retracement` is gone;
  `min_decline` and `min_peak_age_days` are new. The defaults are the spec's:
  `min_decline=0.08`, `min_peak_age_days=20`, `day0_window=5`, `ftd_min_gain=0.0125`,
  `ftd_min_days=4`, `ftd_max_days=25`. `strategies/ftd_belowMA.yaml` was updated to match;
  other strategy files naming the old parameters will fail to build.
- The signal's CSV now carries `peak`, `peak_date`, `decline_pct`, `day0_date`, `rally_low`,
  `rally_day` and `ftd_date` next to `es` and the state, for checking a run by hand.

`test/integration/test_ftd_qqq.py` runs the signal over the QQQ data in `md/` and checks the
Follow Through Days it finds against the reference dates in `docs/SIGNALS.md`.

# September 14, 2026

Added `backfire/visualize.py`, a Plotly Dash dashboard that reads the CSV files a backtest run
wrote into its output folder and serves an interactive view over them - it never re-runs the
backtest. The dashboard has a header with the strategy, the backtest period, the total return,
the CAGR and the three largest drawdowns; a chart pane plotting the strategy's daily balance, the
underlying OHLC, the entry/exit signal regimes as translucent bands and the executed trades as
markers; and a monthly/annual returns table where clicking a cell highlights the start and end of
that period on the chart. Launch with `uv run python backfire/visualize.py -out <run folder>` and
open http://127.0.0.1:8050/.

# May 26, 2025

Backfire backtests strategies consisting of Entry and Exit signals and common risk management and 
position management rules. 

The backtest is performed on daily OHLCV data and emulates the action of a trader who evaluates the market state 
in the evening, decides on actions and then executes the actions in the morning. 

Entry signal: 
Binary signal (True/False) that triggers either once (e.g., price crosses MA to the upside) or stays on 
(e.g., price above MA). When the signal triggers, the strategy establishes position. The same 
signal will not be reentered - i.e., if a position was exited (based on SL, TP, TS or ExitSignal), the position will 
not be reentered until a new entry signal is triggered. A new signal will also not be entered as long as the exit 
signal which caused it to be exited is in place. 
Note that entry signal going from True to False does not mean exiting - only exit signal triggers position exit. Use
negation of entry signal as exit signal if you need such behaviour. 

Exit signal: 
Binary signal that can trigger once or stay on. When it triggers, it takes precedence over the entry signal and full 
position is exited. When it goes off, the position may not be reentered on the same entry signal (strategy needs a new 
entry signal to put on a position) and may not be reentered as long as an exit signal is on.

Risk management: 
Backfire supports stop loss, take profit and trailing stop (based on MA), all specified as percentages
of the underlying. If either of the three is triggered, the current position is closed in full and will not be
reentered until a new entry signal is generated. 

Position management: 
Fixed Percentage: invest a percentage of the portfolio equity on entry, sell the entire positon on exit.  
Gradual Exposure: as long as the same entry signal remains in place, the position size is increased as the 
underlying appreciates based on a predefined schedule, e.g. 10% initially, add 5% on 5% up and add 5% on 10% up.
The position will NOT be decreased as the underlying drops (exit is triggered by SL, TP, TS or exit signal). 

Memo on entry: 
   bought:xxx shares:signal_name; 
Memo on exit: 
   sold:xxx shares:SL/TP/TS/exit_signal_name

Notes: 
I am 
not sure how adding and reducing positions will be handed here, for now the signal is simply True and False. 






# Nov 22, 2022
Use Cases: 
- Utilities for market data maintenance. This includes 'backfire.py get AAPL' and 'backfire.py update_all'.  
- Signal visualization. Visualize signals such as Pocket Pivot, Buyable Gap Up, Distribution Days, Follow-thru Day etc. 
  This will include coding up a subclass of Signal. The Signal is an object returning id (sequential id) and value (-1 .. 1). 
  Signals are visualized by running the VisualizeSignal notebook - the notebook will graph out the signal along with the underlying.  


# Nov 7, 2022
Three work items: 
- don't enter the signal once it has been stopped out.
- make sure exit and risk mgmt signals override the entry signal
- implement partial positions based on strength of signal

## Motivation

One approach to investing in growth stocks is to identify stocks most likely to go up, buy them as they are beginning
their run and before they get overextended, and then sell them into strength or on the growth dissipating. Such 
strategies generally work best in market uptrends. 

Several such strategies are described in books by Zweig, O'Neil, Weinstein, Minervini, Boucher and others - they select
stocks based on the combination of price and volume action (breaking out into an uptrend) and fundamental growth factors 
(e.g., accelerating growth). The entry and exit points are then driven by technical factors - enter on breakout from 
consolidation or other technical setups, sell on signs of weakness or overextension.

These strategies rely on a market indicator which detects the state of the market (uptrend, downtrend, rangebound). These
indicators are based on price and volume of the market action. The strategies are generally fully invested in uptrends 
and in cash during downtrends. 

I want to be able to backtest the performance of these strategies in different market periods.  

Backfire is a backtesting framework which will help me answwer the following questions:  
- do the signals and strategies described in the above books really work when backtested in realistic scenarios? 
- is it better to concentrate on shorter runups and trade more frequently (Minervini-style) or on larger runups
    and hold for longer periods of time (CANSLIM-style)? 
- what happens if I vary strategy parameters (e.g., take profit threshold, signal parameters)? What set of 
    parameters is compatible with CIBC 15 day mandatory holding period?
- how do different market indicators perform? 

Aspirationally, I find the best indicators and parameters, I would like to have a screening tool which would, 
on a weekly basis, allow me to identify stocks as they become candidates for entry and exit.  

The goal would be: 
    - code up a backtesting framework simulating a real-life strategy (daily data, exit as per books)
    - code up indicators used in the above strategies - CANSLIM indicators, Minervini indicators, Boucher 
    - run backtests on all software and technology stocks to determine how plausible these strategies are 
    - code up an automated screening tool identifying stocks for the best strategy

## Description

Backfire is a stock trading strategy backtesting framework. The framework simulates actions of a trader who uses daily data and 
looks at the markets twice: in the evening, the trader evaluates the market situation and decides on the trading 
actions next day, and then executes the trading actions next morning at the opening prices.

The simulated trading strategy uses two signals - entry and exit signals - and supports configurable risk management 
and position management logic. The strategy enters a position whenever the entry signal is triggered (signal is evaluated 
at the end of the day, trading action is executed the next morning at the opening prices) and exits the position whenever 
either the exit or the risk management signals are triggered (again, both exit and risk management signals are 
evaluated at the end of the day and sell actions are executed the next morning). The strategy generates 
data for visualization of its behaviour and signals, as well as the set of trades and trading performance statistic.

## Supported Scenarios

Strategy has: 
- entry signal (evaluated at end of day)
- exit signal (evaluated at end of day) 
- risk management (exit position due to loss exceeding stop loss)
- position management (fixed amount, fixed proportion) 
- market situation overlay 

Strategy can handle the following scenarios: 
- O'Neill CANSLIM trade with simple entry/exit. Entry on pattern (entry signal fires once) but only when the market is on (overlay stays on and off, so 
  need index data as well), exit on one of o/e or weakness (exit signal fires once). Risk mgmt fires on loss > 7% (fires once). 
  If risk mgmt and entry clash, entry wins. 
- Buy and Hold a ticker. entry=AlwaysOn or a signal that fires one, exit=AlwaysOff, rm=None, pos_mgmt=fixed_amount, overlay=None
- Hold Index with Risk Overlay. 
- Buy/Sell based on a single signal. 
- Buy/Sell based on a single signal, with risk management overlay. 
- ONeil with pyramiding.   
- Minervini trade. Splitting exits, 
- 

And so: 
- Entry signal that can fire once
- 





Statistics: cagr, max DD, p, avgW, avgL, avgHpW, avgHpL, #trades
Visualization: equity curve (with underlying), trade histogram, time chart of entry and exit sigansl (with underlying) 





