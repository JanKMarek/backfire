from backfire.base import Environment, NoRiskManagement, SignalDrivenStrategy, AlwaysOnSignal

import os
print(os.getcwd())

ticker = "QQQ"
name = "Index_BuyAndHold"
from_date = "2000-01-01"
out_dir = r"../out/index_buy_hold"

env = Environment(md=r"../md", out_dir=out_dir)
entry_signal = AlwaysOnSignal()
exit_signal = None
risk_management = NoRiskManagement()
# position management - fixed_amount

s = SignalDrivenStrategy(
    env=env,
    entry_signal=entry_signal,
    exit_signal=exit_signal,
    risk_management=risk_management,
    name=name)

s.backtest(ticker=ticker, from_date=from_date)

print("Backtest done.")