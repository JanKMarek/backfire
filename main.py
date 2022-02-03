from backfire import Environment, Runner, Evaluator
from backfire import RandomEntrySignal
from backfire import Strategy

if __name__ == '__main__':

    print("Backtesting TestStrategy")

    r = Runner(Environment(md=r"../md/daily"),
               RandomEntrySignal(name="rand_entry", prob=0.2),
               Strategy(sl=0.07, tp=0.2, pos=100))
    for ticker in ['aapl', 'docu', 'crwd']:
        trades, price_and_signal = r.backtest(ticker=ticker, from_date='2020-01-01')
        price_and_signal.to_csv(f"{ticker}_pas.csv", index=True, header=True)
        stats, trades = Evaluator().evaluate_trades(trades)
        stats.to_csv("stats.csv", header=False, index=True)
        trades.to_csv("trades.csv", header=True, index=False)
        print(f"Stats for {ticker}")
        print(stats)



