from backfire.base import Environment, BasicRiskManagement, SignalDrivenStrategy, PositionManagement
from backfire.signals import FTDSignal, BreakBelowMA

short_MA = 21 # days
long_MA = 120 # days

ftd_min_gain = 0.02
ftd_rally_attempt_min_days = 4
exit_ma_period = 50
stop_loss = None
take_profit = None
trailing_stop_period = None

risk_management = BasicRiskManagement(stop_loss=stop_loss,
                                      take_profit=take_profit,
                                      trailing_stop_period=trailing_stop_period)
ticker = 'QQQ'
name = f"FTDvsBB50"
from_date = '2000-01-01'
out_dir = f"../out/{name}"
md = "../md"

env = Environment(md=md, out_dir=out_dir)
entry_signal = FTDSignal(ftd_min_gain=ftd_min_gain, rally_attempt_min_days=ftd_rally_attempt_min_days)
exit_signal = BreakBelowMA(period=exit_ma_period)

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