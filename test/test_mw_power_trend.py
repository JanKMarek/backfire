from backfire.base import Environment, BasicRiskManagement, SignalDrivenStrategy, PositionManagement, ReverseSignal
from backfire.signals import MWPowerTrend

short_MA = 21 # days
long_MA = 50 # days

stop_loss = None
take_profit = None
trailing_stop_period = None

risk_management = BasicRiskManagement(stop_loss=stop_loss,
                                      take_profit=take_profit,
                                      trailing_stop_period=trailing_stop_period)
ticker = 'QQQ'
name = f"MWPowerTrend"
from_date = '2000-01-01'
out_dir = f"../out/{name}"
md = "../md"

env = Environment(md=md, out_dir=out_dir)
entry_signal = MWPowerTrend(short_MA=short_MA, long_MA=long_MA)
exit_signal = ReverseSignal(entry_signal)

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