Usecases: 

Visualize market signals, e.g., MACrossover, WonDD, WonFTD. Zoom-in, zoom-out.  
Backtest market signals. Analyze individual trades (entry, exit, signal values at entry and exit) in tabular and visual form, produce summary statistics. 
Visualize individual stock signals, e.g., WonBreakout, PP, BGU, etc. 
Backtest individual stock signals. Entry signal with different exit signals - some based on stock behaviour, others on general PnL/retracement principles. 


1./ Visualize market signals. 
Signal is evaluated based on OHLCV at the end of the day. It produces a value between -1 and 1. (short to long). 
A signal may be discrete or continuous. For discrete signals, a signal id is produced. Or this can be kept in the backtest framework.

For example: 
MACrossover: short MA above/below long MA. If above, signal=+1, if below, signal = -1. Each change of discrete signal has a new value. 
MACrossover3: Pr, ShortMA, LongMA. Matrix of six states: 
ShortMA above LongMA: Price > Short: +1; Price between Short and Long: 0; Price below Short and Long: -1. 
ShortMA below LongMA: Price > Long: 1; Price between Short and Long: 0; Price below Short: -1. 
Koteshwar: 
SW: Stan Weinstein