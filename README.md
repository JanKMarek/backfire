# Backfire 

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

## Worklist

To-do list: 
- code backtest tool supporting entry and exit indicators, risk management with trailing stop and fixed positon size
- code MarketState indicators, plus CupAndHandle, PullbackBreakout, FlatBaseBreakout, Breakdown indicators
- backtest O'Neil strategy on the set of software/technology stocks 
- code Minervini's Stage2, VCP and Minervini breakout, Violations exit strategies
- backtest Minervini's strategies

## Issues
- SDS: when an entry signal is used in conjunction with risk management, risk mgmt may stop a position out yet 
  reenter on the next day since the entry signal is still in force. 
- Add Nasdaq data on the secondary axis for relative strength calculations? 
- relative strenth - w.r.t nasdaq

## Miscellaneous
- run jupyter notebook in conda environment 'investing'



