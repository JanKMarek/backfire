# Backfire - quantitative stock investing strategy research engine. 

## Setup
```
uv sync                                    # create .venv and install everything (incl. dev group)
```

## Project Goals and Overview
Backfire is a quantitative stock investing strategy research engine. 

Agents, visualization notebooks and dashboards use common tools and their results to backtest, analyze and optimize stock investing and trading strategies.

The goal is answering questions such as: 
- what risk, return and individual trades would a strategy generate from Jan 2019 to July 2026?
- analyze the results: when/how did the strategy make money/lose money (few/many winning trades vs. many/few losing trades, one big win/loss driving the results) trade holding periods.
- would changing strategy parameter or rule result in better or worse outcome? 
- what is the optimal set of parameters for this strategy? 
- what are the weaknesses of this strategy, e.g. dependency on one big winning trade, sensitivity to slippage (trading at opening price)
- how should this strategy be modified (parameters, rules of behaviour) to improve its performance? 
- what risks are not mitigated in this strategy? 
- how would a strategy behave on a synthetic/simulated market data scenario

The answers are based on quantified strategy risk and return metrics and simulation/backtest result analysis.  

## Application Domain 

### What is a strategy
A stock investing strategy is a set of rules for buying and selling an underlying stock market instrument(s).

The strategy logic simulates actions of a trader who checks the state of the market at the end of the day (and so has access to open, close, high and low prices of the underlyings for the current day), decides on actions (buy, sell, do nothing) and executes the actions at the beginning of the next day (executing at the Open price of the underlying the next day). 

Underlying stock market instruments: 
- index/sector ETFs (e.g., SPY, TQQQ, SOXL)
- individual stocks (e.g., MU, ANET)

Strategy rules: 
- **general overlay rules**: indicates strategy is inactive (a.k.a 'in cash', 'out of the market') or active (i.e., has positions in the market). General overlay rules may be market related (market in downtrend, market choppy) or strategy related (cool-off period after a period of losing trades)
- **selection rules**: what underlying should the strategy buy/sell. 
- **buy rules**: triggers that cause the strategy to buy the underlying. Examples: entry buy rule to establish the initial position in an underlying, add-on rule to add to a winning position. 
- **sell rules**: triggers causing the strategy to sell the underlying. Examples of these are initial stoploss rules to limit losses on losing trades or profit taking rules to take profit in a trade (selling into strength while the underlying is still advancing or selling into weakness after the underlying has begun to decline)
- **position size rules**: determine how much of the underlying (e.g., the number of shares or total dollar position size) should the strategy buy or sell.  

We focus on long-only strategies (no shorting the underlying) and strategies making decisions based on end of the day prices (so no intraday trading). 

All underlyings will be denominated in US Dollars (USD). 

Here is an example of a strategy: 
Strategy "Index 50d/200d MA Crossover". 
- general overlay: none
- selection rules: always buy/sell QQQ
- buy rules: when 50d moving average (MA) crosses above 200d MA from below, buy QQQ
- sell rules: when 50d MA drops below 200d MA, sell QQQ
- position size rules: always buy/sell as much shares as you can (so spend all portfolio cash) 

Another example of a strategy: 
Strategy "CANSLIM Market Leader Breakout, Single Run": 
- general overlay: general market in uptrend 
- selection rules: select most promising market leader breaking out from a consolidation
- buy rules: move above the pivot point on volume
- sell rules: 
  - initial stop loss: 7% below pivot point
  - take profit rule (aka sell into strength): trade profit 25%+ and stock price 10%+ over the 21d MA
  - retracement sell rule (aka sell on weakness): stock price closes below 21d MA
- position size rules: buy shares for 20% of the portfolio equity

And yet another example: 
Strategy "Vibha Jha TQQQ index strategy": 
- general overlay: none
- selection rules: always buy/sell TQQQ
- buy rules: on Follow-Through Day or three Higher Highs/Higher Lows after QQQ consolidation of 8%-10% over at least four weeks
- sell rules: 
  - initial stop loss: QQQ closes below the First Day of Rally low of the day
  - strength take profit rule: trade profit 25%+ AND QQQ 10% over the 21d MA
  - weakness take profit rule: QQQ closes below 21d MA for two days in a row
- position size: 
  - buy TQQQ shares for 50% of the portfolio equity


### Strategy evaluation

Strategy evaluation is the process of comparing strategy performance to its objectives and constraints and identifying any red flags: 

Strategy Objective: 
  - total return over trading period
  - CAGR (compounded annualized growth rate)
  - Calmar ratio (CAGR / MaxDrawdown)
  - Sharpe ratio

Constraints: any of the following 
- maximum drawdown (realized, unrealized) 
- time in drawdown
- number of trades or average trade holding period

Identifying red flags: 
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

### Strategy Structure
To simplify reasoning about strategies and strategy analysis, it is useful to construct a strategy from signals and common rules:  
- **general market overlay signal**: signal calculated from general market averages (e.g., SPY or QQQ indexes) which indicates the level of exposure the strategy should be taking on.
- **entry signal**: Binary signal (True/False) that triggers either once (e.g., price crosses MA to the upside) or stays on (e.g., price above MA). When the signal triggers, the strategy establishes position. The same signal will not be reentered - i.e., if a position was exited (based on SL, TP, TS or ExitSignal), the position will not be reentered until a new entry signal is triggered. A new signal will also not be entered as long as the exit signal which caused it to be exited is in place. Note that entry signal going from True to False does not mean exiting - only exit signal triggers position exit. Use negation of entry signal as exit signal if you need such behaviour. 
- **exit signal**: Binary signal that can trigger once or stay on. When it triggers, it takes precedence over the entry signal and full position is exited. When it goes off, the position may not be reentered on the same entry signal (strategy needs a new entry signal to put on a position) and may not be reentered as long as an exit signal is on.
- **risk management rules**: 
  - stop loss rule: when in position and trade loss exceeds a threshold, close the position and do not act until a new entry signal is triggered
  - trailing stop rule: when price falls back below a threshold (e.g., 21d MA or -10%), close the position and do not act until a new entry signal is triggered
  - take profit rule: when in position and trade gain exceeds a threshold, close the position and do not act antil a new entry signal is triggered
- **position size management rules**: 
    - Fixed Fraction: invest a percentage of the portfolio equity on entry, sell the entire positon on exit.  
    - Gradual Exposure: as long as the same entry signal remains in place, the position size is increased as the underlying appreciates based on a predefined schedule, e.g. 10% initially, add 5% on 5% up and add 5% on 10% up. The position will NOT be decreased as the underlying drops (exit is triggered by SL, TP, TS or exit signal). 

All signals have values between 0 (no signal) to 1 (maximum signal strength). 


### Simulating/backtesting strategy execution 

The strategy simulation takes the following parameters: 
  - start date, end date
  - underlying (ticker)
  - strategy definition (yaml file with command line overrides)
  - out : persistent output folder ("" means no persistent output)

In the simulation, the strategy starts with a certain portfolio size in cash, executes its rules over the trading period and closes all positions at the end of the trading period.  

The simulation framework generates performance statistics, list of trades, daily file and signal files: 

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

Daily file - for every day in the trading period: 
- Date
- O,H,L,C,V prices
- es,es_id,xs,pos,cash,action,delta_shares,memo,buy_price,trailing_stop,balance,unrealized_CumMax,unrealized_dd,unrealized_dd_pcnt

Signal files - for each signal a file with one row for every day in the trading period: 
- Date
- signal value 
- signal id


All dates in YYYY-MM-DD. All stock prices reported with two decimal places. All Pnl/gain/loss numbers with zero decimal places and comma at thousands. All percentages with two decimal places, e.g. 0.77 or -0.33. 


### Market data 
Market data is provided in the OHLCV format - open, high, low, close and volume. 




## Architecture and Design decisions

### Components and Interfaces

- load_ohlcv: This module obtains data from the data source (currently yahoo finance only), caches it for later use and returns it in a dataframe with defined columns. 

- Strategy composed of signals is implemented as class SignalDrivenStrategy. An instance of this class is configured with appropriate instances of entry and exit signals and risk management and position size rules. Signals are implemented as subclasses of class Signal. Each signal value is given a signal id which is important when we do not want to act on the same signal once stopped out, for instance. Common risk management rules are implemented in BasicRiskManagement and position management rules in PositionManagement. The strategy's _run method generates two tables - one with trades generated, the other one with prices, signals and equity curve. 

- Evaluator class computes evaluation metrics.  

- Method Strategy.backtest encapsulates the entire process of a backtest run - load market data, generate signals, generate trades, evaluate statistics, save evaluation data to experiment output files and return evaluation data to the caller. 

- Visualization tools use Jupyter notebooks or Plotly dashboards and consume the evaluation data generated by the Strategy.backtest method. 

- AI agents use skills which encapsulate the load_ohlcv and the Strategy.backtest methods. 

- Optimization utilities use the load_ohlcv and Strategy.backtest methods. 


## CLI Workflows

- Backtesting a strategy: 

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
    take_profit: 0.25
    trailing_stop_period: null
  position_management: 
    name: PositionManagement
    initial_position: 100000
    policy: fixed_fraction
    fraction: 1.0      
```
`uv run python src/backfire/backtest.py --start_date "2020-01-01" --end_date "2026-07-01" -md "md/daily" --underlying QQQ --strategy my_strategy.yaml --out "test/my_strategy"`

- Running visualizatino notebooks: 
`uv run jupyter lab`

- Run fast unit tests
`uv run pytest tests/unit -q`

- Run full test suite
`uv run pytest tests -q`







