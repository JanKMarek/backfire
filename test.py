from backfire import Environment, Signal
from backfire import SignalDrivenStrategy
from datetime import datetime
import pandas as pd

t = """
Day       ,  O,  H,  L,  C,  V,es,pos, sl, tp,action,memo  ,notes
2021-06-06,100,103, 98,101,  1, f,  0,   ,   ,      ,      , 
2021-06-07,110,113,108,111,  1, f,  0,   ,   ,      ,      ,
2021-06-08,120,123,118,121,  1, T,  0,   ,   ,buy   ,      ,signal triggered based on Closing price; buy next day at price O
2021-06-09,130,133,128,131,  1, f,  1,121,156,so    ,bought,signal off; sell next day at price O
2021-06-10,140,143,138,141,  1, f,  0,   ,   ,      ,sold  , 
2021-06-11,150,153,114,118,  1, T,  0,   ,   ,buy   ,      ,signal triggered, by next day 
2021-06-12,160,163,158,161,  1, T,  1,149,192,      ,bought, 
2021-06-13,170,173,144,144,  1, T,  1,149,192,sl    ,      ,stoploss triggered, signal still on
2021-06-14,180,183,178,181,  1, T,  0,   ,   ,buy   ,      ,signal triggered based on Closing price; buy next day at price O
2021-06-15,190,193,188,191,  1, T,  1,177,228,      ,bought, 
2021-06-16,200,203,198,201,  1, T,  1,177,228,      ,      ,
2021-06-17,210,235,208,230,  1, f,  1,177,228,tp    ,      ,tp; closing price > tp; sell next day at price O
2021-06-18,220,223,218,221,  1, f,  0,   ,   ,      ,sold  , 
2021-06-19,230,233,228,231,  1, f,  0,   ,   ,      ,      ,
2021-06-20,240,243,238,241,  1, T,  0,   ,   ,buy   ,      ,signal triggered based on Closing price; buy next day at price O
2021-06-21,250,253,227,251,  1, T,  1,233,300,      ,bought,
2021-06-22,260,263,258,261,  1, f,  0,233,300,so    ,      ,signal off
2021-06-23,270,273,268,271,  1, f,  0,   ,   ,      ,sold  ,
2021-06-24,280,283,278,281,  1, T,  0,   ,   ,buy   ,      ,signal triggered based on Closing price; buy next day at price O
2021-06-25,290,355,288,350,  1, T,  1,270,348,tp    ,bought,tp; closing price > tp; sell next day at price O
2021-06-26,300,303,298,301,  1, f,  0,   ,   ,      ,sold  ,     
2021-06-27,310,313,308,311,  1, f,  0,   ,   ,      ,      ,
2021-06-28,320,323,318,321,  1, T,  0,   ,   ,buy   ,      ,
2021-06-29,330,333,328,331,  1, T,  1,307,396,      ,bought,signal triggered based on Closing price; buy next day at price O
2021-06-30,340,343,338,341,  1, T,  1,307,396,      ,      ,
2021-07-01,350,353,348,351,  1, T,  1,307,396,fday  ,sold  ,last day trade; sell at closing price if position not zero"""

trades = """
ticker,entry_date,shares,entry_price, exit_date,exit_price,memo
      ,2021-06-09,    769, 130,2021-06-10,       140,   
      ,2021-06-12,    625, 160,2021-06-14,       180,
      ,2021-06-15,    526, 190,2021-06-18,       220,         
      ,2021-06-21,    400, 250,2021-06-23,       270,
      ,2021-06-29,    345, 290,2021-06-27,       310,         
      ,2021-06-29,    303, 330,2021-07-01,       350,   
"""

def make_md(t):
    ndx = []
    data = []
    for line in t.splitlines():
        if line[:3] == "Day" or line == '':
            continue
        fields = line.split(",")
        ndx.append(datetime.strptime(fields[0], "%Y-%m-%d").date())
        data.append(fields[1:6])
    rv = pd.DataFrame(data=data, index=ndx, columns=['O','H','L','C','V'])
    for column in rv.columns:
        rv[column] = pd.to_numeric(rv[column], downcast="float")
    return rv

class TestSignal(Signal):
    def __call__(self, ohlcv):
        rv = ohlcv.copy()
        rv['es'] = False
        def f(s):
            return datetime.strptime(s, "%Y-%m-%d").date()
        rv.loc[f("2021-06-08"), 'es'] = True
        rv.loc[f("2021-06-11"), 'es'] = True
        rv.loc[f("2021-06-12"), 'es'] = True
        rv.loc[f("2021-06-13"), 'es'] = True
        rv.loc[f("2021-06-14"), 'es'] = True
        rv.loc[f("2021-06-15"), 'es'] = True
        rv.loc[f("2021-06-16"), 'es'] = True
        rv.loc[f("2021-06-20"), 'es'] = True
        rv.loc[f("2021-06-21"), 'es'] = True
        rv.loc[f("2021-06-24"), 'es'] = True
        rv.loc[f("2021-06-25"), 'es'] = True
        rv.loc[f("2021-06-28"), 'es'] = True
        rv.loc[f("2021-06-29"), 'es'] = True
        rv.loc[f("2021-06-30"), 'es'] = True
        rv.loc[f("2021-07-01"), 'es'] = True
        rv = rv.es
        rv.rename('es', inplace=True)
        return rv

if __name__ == '__main__':

    md = make_md(t)

    env = Environment(md=md, out_dir=r"./out_unittest")
    s = SignalDrivenStrategy(
        env,
        signal=TestSignal("testsignal"),
        sl=0.07,
        tp=0.2,
        pos_management="fixed"
    )

    pas, positions, trades = s.backtest(ticker="", from_date='2021-06-06')
    print("done")