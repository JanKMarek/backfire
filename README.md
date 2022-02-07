# Backfire 

## Introduction


I want to gain a deeper understanding of growth-oriented stock trading strategies - strategies where we identify stocks most
likely to grow over the next few months, enter them before they get overextended and sell them before 
they crash. I want to research how to identify such companies (by a combination of fundamental and technical factors), 
when to enter them and when to exit them. 
   
Several such strategies are described in books by Zweig, O'Neil, Minervini, Boucher and others - they tend to select
stocks based on relative strength, developing uptrend and fundamental growth factors (e.g., accelerating earnings growth). 
The entry and exit points are then driven by technical factors - enter on breakout from consolidation, sell on signs of weakness 
or overextension. These strategies then differ in the entry signals used and in the size of the uptrends they attempt 
to capture. 

I want to have my own backtest tools which will allow me to decide on questions such as: 
- do the signals and strategies described in the above books really work when backtested in realistic scenarios? 
- is it better to concentrate on shorter runups and trade more frequently (Minervini-style) or on larger runups
    and hold for longer periods of time (CANSLIM-style)? 
- what happens if I vary strategy parameters (e.g., take profit threshold, signal parameters)? What set of 
    parameters is compatible with CIBC 15 day mandatory holding period?
        
Once I identify strategies and parameters that would work (i.e., have desired risk, reward and are compatible with
with CIBC code of conduct), I would like to have a screening tool which would, on a weekly basis, allow me to identify stocks 
as they become candidates for entry and exit.  

The goal would be: 
    - code up a backtesting framework simulating a real-life strategy (daily data, exit as per books)
    - code up indicators used in the above strategies - CANSLIM indicators, Minervini indicators, Boucher 
    - run backtests on all software and technology stocks to determine how plausible these strategies are 
    - code up an automated screening tool identifying stocks for the best strategy

## Description

Backfire is a stock trading strategy backtesting framework. The framework simulates a trader who uses daily data and 
looks at the markets twice - in the evening, the trader evaluates the market situation and decides on the trading 
actions next day, and then executes the tradign actinos next morning at the opening prices.

The simulated trading strategy uses two signals - entry and exit signals - and supports configurable risk management 
and position management logic. The strategy enters a position whenever the entry signal is triggered 
and exits the position whenever either the exit or the risk management signals are triggered. The strategy generates 
data for visualization of its behaviour and signals, as well as the set of trades and trading performance statistic. 

## Worklist

To-do list: 
- code backtest tool supporting entry and exit indicators, risk management with trailing stop and fixed positon size
- code MarketState indicators, plus CupAndHandle, PullbackBreakout, FlatBaseBreakout, Breakdown indicators
- backtest O'Neil strategy on the set of software/technology stocks 
- code Minervini's Stage2, VCP and Minervini breakout, Violations exit strategies
- backtest Minervini's strategies



