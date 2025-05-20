from backfire.base import Environment, BasicRiskManagement, PositionManagement, SignalDrivenStrategy
from backfire.signals import ShortMABelowLongMA, ShortMAAboveLongMA

short_MA = 50  # days
long_MA = 200  # days
stop_loss = None
take_profit = 0.6
trailing_stop_period = 200

ticker = 'QQQ'  # '^IXIC_1990'
name = f"Index_{str(short_MA)}dMAvs{str(long_MA)}dMA"
from_date = '2000-01-01'
out_dir = f"../out/{name}"
md = "../md"



env = Environment(md=md, out_dir=out_dir)

entry_signal = ShortMAAboveLongMA(short_MA=short_MA, long_MA=long_MA)
exit_signal = ShortMABelowLongMA(short_MA=short_MA, long_MA=long_MA)
# risk_management = NoRiskManagement()
risk_management = BasicRiskManagement(
    stop_loss=stop_loss,
    take_profit=take_profit,
    trailing_stop_period=trailing_stop_period)
position_management = PositionManagement(initial_position=100000, policy="fixed_fraction", fraction=1.0)

s = SignalDrivenStrategy(
    env=env,
    entry_signal=entry_signal,
    exit_signal=exit_signal,
    risk_management=risk_management,
    position_management=position_management,
    name=name)

s.backtest(ticker=ticker, from_date=from_date)