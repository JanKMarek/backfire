from backfire.base import Environment, BasicRiskManagement, SignalDrivenStrategy, GradualExposure
from backfire.signals import ShortMAAboveLongMA, ShortMABelowLongMA

short_MA = 50 # days
long_MA = 200 # days

stop_loss = 0.07
take_profit = 0.25
trailing_stop_period = None

risk_management = BasicRiskManagement(stop_loss=stop_loss,
                                      take_profit=take_profit,
                                      trailing_stop_period=trailing_stop_period)
ticker = 'QQQ'
name = f"ShortMAVsLongMA"
from_date = '2000-01-01'
out_dir = f"../out/{name}"
md = "../md"

env = Environment(md=md, out_dir=out_dir)
entry_signal = ShortMAAboveLongMA(short_MA=short_MA, long_MA=long_MA)
exit_signal = ShortMABelowLongMA(short_MA=short_MA, long_MA=long_MA)

position_management = GradualExposure(
    initial_position=100000,
    schedule={
        0: 0.1,
        0.03: 0.05,
        0.05: 0.05
    })

s = SignalDrivenStrategy(
        env=env,
        entry_signal=entry_signal,
        exit_signal=exit_signal,
        risk_management=risk_management,
        position_management=position_management,
        name=name)
s.backtest(ticker=ticker, from_date=from_date)

print("Backtest done.")