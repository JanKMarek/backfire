from backfire.base import Environment, BasicRiskManagement, SignalDrivenStrategy, PositionManagement
from backfire.signals import OrSignal, ShortMAAboveLongMA, ShortMABelowLongMA, TakeProfitSignal

short_MA = 50 # days
long_MA = 200 # days
stop_loss = 0.07
take_profit = 0.25

risk_management = BasicRiskManagement(stop_loss=stop_loss)
ticker = 'QQQ'
name = f"ShortMAVsLongMA"
from_date = '2000-01-01'
out_dir = f"./out/test/{name}"
md = "./md"

env = Environment(md=md, out_dir=out_dir)
entry_signal = ShortMAAboveLongMA(short_MA=short_MA, long_MA=long_MA)
exit_signal = OrSignal(ShortMABelowLongMA(short_MA=short_MA, long_MA=long_MA),
                       TakeProfitSignal(threshold=take_profit))

position_management = PositionManagement(initial_position=100000, policy="fixed_fraction", fraction=1.0)

s = SignalDrivenStrategy(
        env=env,
        entry_signal=entry_signal,
        exit_signal=exit_signal,
        risk_management=risk_management,
        position_management=position_management,
        name=name)
s.backtest(ticker=ticker, from_date=from_date)

print("Backtest done.")