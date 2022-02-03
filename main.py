from backfire import Environment, Runner, Evaluator
from backfire import RandomEntrySignal, FlatBaseSignal
from backfire import Strategy

if __name__ == '__main__':

    print("Backtesting TestStrategy")

    r = Runner(Environment(md=r"../md/daily", out_dir=r"./out_testing"),
#               RandomEntrySignal(prob=0.2),
               FlatBaseSignal(),
               Strategy(sl=0.07, tp=0.2, pos=100))

    for ticker in ['aapl', 'docu', 'crwd']:
        trades, price_and_signal = r.backtest(ticker=ticker, from_date='2020-01-01')



