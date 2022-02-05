import backfire as bf

if __name__ == '__main__':

    # s = BuyAndHoldStrategy(
    #     env=Environment(md=r"../md/daily", out_dir=r"./out_test"))
    #
    # s.backtest(ticker="AAPL", from_date="2020-01-01")

    # env = bf.Environment(md=r"../md/daily", out_dir=r"./out_test")
    # s = bf.SignalDrivenStrategy(
    #     env,
    #     signal=bf.FlatBaseBreakoutSignal(),
    #     sl=0.07,
    #     tp=0.2,
    #     pos_management="fixed"
    # )

    env = bf.Environment(md=r"../md/daily", out_dir=r"./out_test")
    s = bf.SignalDrivenStrategy(
        env,
        signal=bf.Stage2Signal(),
        sl=0.07,
        tp=5.0,
        pos_management="fixed"
    )

    s.backtest(ticker="FSLY", from_date='2016-01-01')




