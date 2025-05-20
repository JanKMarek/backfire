import backfire as bf

if __name__ == '__main__':

    # OHLCV records for a ticker
    env = bf.Environment(md=r"../md/daily", out_dir=r"./out_test")

    # SignalDriveStrategy, standard configuration (7% stoploss, follow w/10d ma after 20%)
    s = bf.SignalDrivenStrategy(
        env,
        entry_signal=bf.Stage2Signal(),
#        entry_signal=[bf.Stage2Signal(), bf.FlatBaseBreakoutSignal()], # NB need a dynamic way of combining signals
        name="SDS_standard"
    )
    s.backtest(ticker="FSLY", from_date='2016-01-01')

    # SignalDriveStrategy, always take profit at 20%
    s = bf.SignalDrivenStrategy(
        env,
        entry_signal=bf.Stage2Signal(),
#        entry_signal=[bf.Stage2Signal(), bf.FlatBaseBreakoutSignal()], # need a dynamic way of combining signals
        risk_management=bf.OpenProtectiveStop(stop_loss=0.07, take_profit_threshold=.2, trailing_stop_period=0),
        position_management = bf.PositionManagement(initial_position=100000, policy="fixed_amount"), # fixed_fraction,
        name="SDS_alwaysTakeProfit"
    )

    s.backtest(ticker="FSLY", from_date='2016-01-01')

    # BuyAndHoldStrategy
    bh = bf.SignalDrivenStrategy(
        env,
        entry_signal=bf.AlwaysOnSignal(),
        risk_management=bf.NoRiskManagement(),
        name="SDS_BuyAndHold"
    )
    bh.backtest(ticker="FSLY", from_date='2016-01-01')




