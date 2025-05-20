import backfire as bf

if __name__ == '__main__':

    ticker = "FSLY"
    from_date = "2015-01-01"
    out_dir = r"./out_visualize_debug"

    env = bf.Environment(md=r"../md/daily", out_dir=out_dir)
    entry_signal = bf.Stage2Signal()
    exit_signal = None
    risk_management = bf.OpenProtectiveStop(stop_loss=0.07, take_profit_threshold=0.2, trailing_stop_period=20)
    # position management - fixed_amount

    s = bf.SignalDrivenStrategy(
        env=env,
        entry_signal=bf.Stage2Signal(),
        exit_signal=exit_signal,
        risk_management=risk_management,
        name=f"SDS_standard")
    s.backtest(ticker=ticker, from_date=from_date)





