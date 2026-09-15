from datetime import date

import pandas as pd
import pytest
import yaml

from backfire.backtest import (
    apply_overrides,
    build_strategy,
    format_stats,
    load_strategy_conf,
    main,
    run_backtest,
)
from backfire.base import (
    BasicRiskManagement,
    Environment,
    Evaluator,
    NoRiskManagement,
    PositionManagement,
    SignalDrivenStrategy,
)
from backfire.signals import (
    AlwaysOnSignal,
    OrSignal,
    RetracementSignal,
    ShortMAAboveLongMA,
    ShortMABelowLongMA,
    TakeProfitSignal,
    TrailingTakeProfitSignal,
)

STRATEGY = """
strategy:
  name: ShortMAVsLongMA
  entry_signal:
    name: ShortMAAboveLongMA
    short_MA: 50
    long_MA: 200
  exit_signal:
    name: OrSignal
    signals:
      - name: ShortMABelowLongMA
        short_MA: 50
        long_MA: 200
      - name: TakeProfitSignal
        threshold: 0.25
  risk_management:
    name: BasicRiskManagement
    stop_loss: 0.07
  position_management:
    name: PositionManagement
    initial_position: 100000
    policy: fixed_fraction
    fraction: 1.0
"""


@pytest.fixture
def strategy_file(tmp_path):
    path = tmp_path / "strategy.yaml"
    path.write_text(STRATEGY)
    return str(path)


@pytest.fixture
def md(tmp_path):
    """
        Market data directory holding TEST.csv - a rising then falling market, long enough
        for the 50d/200d crossover to trigger both an entry and an exit.
    """
    days = pd.bdate_range("2020-01-01", periods=700).date
    prices = [100.0 + i for i in range(350)] + [450.0 - i for i in range(350)]
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    pd.DataFrame(
        {'Open': prices, 'High': [p + 1 for p in prices], 'Low': [p - 1 for p in prices],
         'Close': prices, 'Volume': [1_000_000.0] * len(prices)},
        index=pd.Index(days, name='Date')).to_csv(md_dir / "TEST.csv")
    return str(md_dir)


def test_build_strategy_instantiates_the_named_components(strategy_file):
    s = build_strategy(load_strategy_conf(strategy_file), Environment(md="md", out_dir=""))

    assert s.name == "ShortMAVsLongMA"
    assert isinstance(s.entry_signal, ShortMAAboveLongMA)
    assert (s.entry_signal.short_MA, s.entry_signal.long_MA) == (50, 200)
    assert isinstance(s.exit_signal, OrSignal)
    assert isinstance(s.exit_signal.signals[0], ShortMABelowLongMA)
    assert isinstance(s.exit_signal.signals[1], TakeProfitSignal)
    assert s.exit_signal.signals[1].threshold == 0.25
    assert isinstance(s.risk_management, BasicRiskManagement)
    assert s.risk_management.stop_loss == 0.07
    assert isinstance(s.position_management, PositionManagement)
    assert s.position_management.initial_position == 100000


@pytest.mark.parametrize("parameter", ["take_profit", "trailing_stop_period"])
def test_take_profit_and_trailing_stop_are_no_longer_risk_management_parameters(parameter):
    conf = {'strategy': {'entry_signal': {'name': 'AlwaysOnSignal'},
                         'risk_management': {'name': 'BasicRiskManagement',
                                             'stop_loss': 0.07, parameter: 10}}}

    with pytest.raises(ValueError, match="Cannot build risk management"):
        build_strategy(conf, Environment(md="md", out_dir=""))


def test_the_stop_loss_fires_below_the_buy_price_by_the_given_fraction():
    rm = BasicRiskManagement(stop_loss=0.07)
    row = lambda close: pd.Series({'C': close, 'buy_price': 100.0})

    assert rm(row(93.5)) is None
    assert rm(row(92.9)) == "sl"


def market_with_closes(closes):
    """
        A market with the given daily closes, the open equal to the previous close, so a
        trade bought on day t+1 pays the close of day t.
    """
    dates = pd.bdate_range("2020-01-01", periods=len(closes)).date
    return pd.DataFrame({'O': [closes[0]] + closes[:-1], 'H': [c + 0.5 for c in closes],
                         'L': [c - 0.5 for c in closes], 'C': closes, 'V': 1_000_000.0},
                        index=dates)


def rising_market(days, start=100.0, step=1.0):
    """
        A market rising by 'step' every day: the closes 100, 101, 102, ...
    """
    return market_with_closes([start + i * step for i in range(days)])


def take_profit_backtest(exit_signal, market):
    strategy = SignalDrivenStrategy(
        env=Environment(md=market, out_dir=""),
        entry_signal=AlwaysOnSignal(),
        exit_signal=exit_signal,
        risk_management=NoRiskManagement(),
        position_management=PositionManagement(initial_position=100_000),
        save_signals=False)
    return strategy.backtest("TEST", from_date=market.index[0], to_date=market.index[-1])


def test_the_take_profit_exit_signal_sells_once_the_trade_gain_exceeds_the_threshold():
    market = rising_market(40)

    _, positions, trades, _ = take_profit_backtest(TakeProfitSignal(threshold=0.1), market)

    assert len(trades) == 1
    trade = trades.iloc[0]
    # bought at the open of the third day (the always on signal is first read on day two),
    # the profit target is crossed on the first close above 110% of that price and the
    # position is sold at the next open
    assert trade.entry_price == 101.0
    first_day_above = next(d for d, c in market.C.items() if c > 101.0 * 1.1)
    assert positions.loc[first_day_above, 'xs'] == market.loc[first_day_above, 'C']
    assert positions.loc[first_day_above, 'action'] == 'sell'
    assert trade.exit_date == market.index[list(market.index).index(first_day_above) + 1]
    assert trade.memo.endswith("TakeProfit_0.1")
    assert trade.pnl_pcnt > 0.1


def test_after_taking_profit_the_same_entry_signal_is_not_re_entered():
    market = rising_market(60)

    _, positions, trades, _ = take_profit_backtest(TakeProfitSignal(threshold=0.1), market)

    # the entry signal stays on for the whole period, yet the strategy stays in cash
    # after selling into strength rather than buying straight back
    assert len(trades) == 1
    assert (positions.pos.iloc[-10:] == 0).all()


def test_the_take_profit_signal_combines_with_a_market_exit_signal():
    market = rising_market(40)
    exit_signal = OrSignal(ShortMABelowLongMA(short_MA=5, long_MA=20),
                           TakeProfitSignal(threshold=0.1))

    _, _, trades, _ = take_profit_backtest(exit_signal, market)

    assert len(trades) == 1
    assert trades.iloc[0].pnl_pcnt > 0.1


def test_without_a_take_profit_the_rising_market_is_held_to_the_last_day():
    market = rising_market(40)

    _, _, trades, _ = take_profit_backtest(None, market)

    assert len(trades) == 1
    assert trades.iloc[0].memo.endswith("lastday")


def test_the_trailing_take_profit_exit_signal_sells_on_the_retracement_after_the_threshold_gain():
    # rises 100 -> 130, then gives back 2 a day: the first down day closes at 128, level
    # with the 5d average of the previous closes; the second, at 126, undercuts it
    up = [100.0 + i for i in range(31)]
    down = [130.0 - 2 * i for i in range(1, 20)]
    market = market_with_closes(up + down)

    _, positions, trades, _ = take_profit_backtest(
        TrailingTakeProfitSignal(threshold=0.1, period=5), market)

    assert len(trades) == 1
    trade = trades.iloc[0]
    trigger_day = market.index[len(up) + 1]                # the second down day
    assert positions.loc[trigger_day, 'action'] == 'sell'
    assert positions.loc[trigger_day, 'xs'] == 126.0
    assert trade.exit_price == 126.0
    assert trade.memo.endswith("TrailingTakeProfit_0.1_5dMA")
    assert trade.pnl_pcnt == pytest.approx(126.0 / 101.0 - 1)
    # the entry signal never turned off, so the strategy stays in cash afterwards
    assert (positions.pos.iloc[-5:] == 0).all()


def test_the_trailing_take_profit_is_not_armed_by_a_retracement_below_the_threshold_gain():
    # peaks 5% up, then falls below the average: no exit, the trade is held to the end
    closes = [100.0 + i * 0.5 for i in range(11)] + [105.0 - i for i in range(1, 10)]
    market = market_with_closes(closes)

    _, _, trades, _ = take_profit_backtest(TrailingTakeProfitSignal(threshold=0.1, period=5), market)

    assert len(trades) == 1
    assert trades.iloc[0].memo.endswith("lastday")


def test_the_trailing_take_profit_is_disarmed_when_a_stop_loss_closes_the_trade():
    # the first trade (bought at 102) arms the trailing take profit at 114, then crashes
    # straight through both the average and the stop loss, which takes precedence; the entry
    # signal turns off and on again so a second trade is taken, which must start un-armed: it
    # dips below the average before its own threshold gain and must be held, not sold
    up = [100.0 + i * 2 for i in range(8)]                 # 100 -> 114, arms at > 112.2
    crash = [90.0]                                         # 7% stop loss at 94.86
    flat = [90.0] * 5
    second = [90.0, 91.0, 92.0, 93.0, 94.0, 90.0, 89.0, 88.0, 87.0, 86.0]
    closes = up + crash + flat + second
    market = market_with_closes(closes)
    entry = pd.Series(False, index=market.index)
    entry.iloc[:10] = True
    entry.iloc[15:] = True

    class Entry(AlwaysOnSignal):
        def _call_impl(self, ohlcv):
            return pd.DataFrame({'es': entry.values}, index=ohlcv.index)

    strategy = SignalDrivenStrategy(
        env=Environment(md=market, out_dir=""),
        entry_signal=Entry(),
        exit_signal=TrailingTakeProfitSignal(threshold=0.1, period=3),
        risk_management=BasicRiskManagement(stop_loss=0.07),
        position_management=PositionManagement(initial_position=100_000),
        save_signals=False)
    _, positions, trades, _ = strategy.backtest("TEST", from_date=market.index[0],
                                                to_date=market.index[-1])

    assert len(trades) == 2
    assert trades.iloc[0].memo.endswith("sl")
    assert trades.iloc[1].memo.endswith("lastday")


def test_the_retracement_exit_signal_sells_once_the_close_falls_the_retracement_below_the_high():
    # rises 100 -> 130 (highs 0.5 above the closes), then gives back 2 a day: 10% below the
    # 130.5 high is 117.45, first undercut by the seventh down day's close of 116
    up = [100.0 + i for i in range(31)]
    down = [130.0 - 2 * i for i in range(1, 20)]
    market = market_with_closes(up + down)

    _, positions, trades, _ = take_profit_backtest(RetracementSignal(retracement=0.1), market)

    assert len(trades) == 1
    trade = trades.iloc[0]
    trigger_day = market.index[len(up) + 6]
    assert positions.loc[trigger_day, 'action'] == 'sell'
    assert positions.loc[trigger_day, 'xs'] == 116.0
    assert positions.loc[market.index[len(up) + 5], 'xs'] == 0
    assert trade.exit_price == 116.0
    assert trade.memo.endswith("Retracement_0.1")


def test_the_retracement_is_measured_from_the_high_since_the_position_was_opened():
    # the market falls from 200 to 150 before the entry signal turns on, then drifts flat:
    # the trade is held, the earlier high predates the position
    fall = [200.0 - 5 * i for i in range(11)]
    flat = [150.0] * 20
    market = market_with_closes(fall + flat)
    entry = pd.Series(False, index=market.index)
    entry.iloc[len(fall):] = True

    class Entry(AlwaysOnSignal):
        def _call_impl(self, ohlcv):
            return pd.DataFrame({'es': entry.values}, index=ohlcv.index)

    strategy = SignalDrivenStrategy(
        env=Environment(md=market, out_dir=""),
        entry_signal=Entry(),
        exit_signal=RetracementSignal(retracement=0.1),
        position_management=PositionManagement(initial_position=100_000),
        save_signals=False)
    _, _, trades, _ = strategy.backtest("TEST", from_date=market.index[0], to_date=market.index[-1])

    assert len(trades) == 1
    assert trades.iloc[0].memo.endswith("lastday")


def test_optional_sections_fall_back_to_the_strategy_defaults():
    conf = {'strategy': {'name': 'EntryOnly',
                         'entry_signal': {'name': 'AlwaysOnSignal'}}}

    s = build_strategy(conf, Environment(md="md", out_dir=""))

    assert s.exit_signal.name == "AlwaysOffSignal"
    assert s.risk_management.name == "NoRiskManagement"
    assert isinstance(s.position_management, PositionManagement)


def test_a_nested_named_mapping_is_built_as_a_signal():
    conf = {'strategy': {'name': 'Reversed',
                         'entry_signal': {'name': 'AlwaysOnSignal'},
                         'exit_signal': {'name': 'ReverseSignal',
                                         'signal': {'name': 'ShortMAAboveLongMA',
                                                    'short_MA': 50, 'long_MA': 200}}}}

    s = build_strategy(conf, Environment(md="md", out_dir=""))

    assert isinstance(s.exit_signal.signal, ShortMAAboveLongMA)


def test_a_list_of_named_mappings_is_built_as_the_signals_of_a_combinator():
    conf = {'strategy': {'name': 'Either',
                         'entry_signal': {'name': 'AlwaysOnSignal'},
                         'exit_signal': {'name': 'OrSignal',
                                         'signals': [{'name': 'ShortMABelowLongMA',
                                                      'short_MA': 50, 'long_MA': 200},
                                                     {'name': 'ReverseSignal',
                                                      'signal': {'name': 'AlwaysOnSignal'}}]}}}

    s = build_strategy(conf, Environment(md="md", out_dir=""))

    assert isinstance(s.exit_signal, OrSignal)
    assert isinstance(s.exit_signal.signals[0], ShortMABelowLongMA)
    assert s.exit_signal.signals[1].name == "ReverseSignal_AlwaysOnSignal"


def test_entry_signal_is_mandatory():
    with pytest.raises(ValueError, match="entry_signal"):
        build_strategy({'strategy': {'name': 'Nothing'}}, Environment(md="md", out_dir=""))


def test_unknown_component_names_the_alternatives():
    conf = {'strategy': {'entry_signal': {'name': 'NoSuchSignal'}}}

    with pytest.raises(ValueError, match="Unknown entry signal 'NoSuchSignal'"):
        build_strategy(conf, Environment(md="md", out_dir=""))


def test_unexpected_component_argument_is_reported():
    conf = {'strategy': {'entry_signal': {'name': 'ShortMAAboveLongMA',
                                          'short_MA': 50, 'long_MA': 200, 'typo': 1}}}

    with pytest.raises(ValueError, match="Cannot build entry signal"):
        build_strategy(conf, Environment(md="md", out_dir=""))


def test_overrides_replace_entries_and_keep_yaml_types(strategy_file):
    conf = load_strategy_conf(strategy_file,
                              ["strategy.entry_signal.short_MA=20",
                               "strategy.risk_management.stop_loss=null",
                               "strategy.name=Tweaked"])

    assert conf['strategy']['entry_signal']['short_MA'] == 20
    assert conf['strategy']['risk_management']['stop_loss'] is None
    assert conf['strategy']['name'] == "Tweaked"
    # untouched entries survive
    assert conf['strategy']['entry_signal']['long_MA'] == 200


def test_overrides_create_missing_sections():
    conf = apply_overrides({}, ["strategy.position_management.fraction=0.5"])

    assert conf == {'strategy': {'position_management': {'fraction': 0.5}}}


@pytest.mark.parametrize("override", ["strategy.entry_signal.short_MA", "=20", "."])
def test_malformed_overrides_are_rejected(override):
    with pytest.raises(ValueError):
        apply_overrides({}, [override])


def test_run_backtest_trades_and_writes_the_run_to_the_output_folder(md, tmp_path):
    out = tmp_path / "run"
    conf = yaml.safe_load(STRATEGY)

    strategy, pas, positions, trades, stats = run_backtest(
        conf, underlying="TEST", start_date="2020-01-01", end_date="2022-12-31",
        md=md, out=str(out))

    assert len(trades) == stats['no_trades'] > 0
    assert (trades.exit_date > trades.entry_date).all()
    assert list(pas.index) == list(positions.index)
    written = {p.name for p in out.iterdir()}
    assert {"pos_TEST_ShortMAVsLongMA.csv",
            "stats_TEST_ShortMAVsLongMA.csv",
            "trades_TEST_ShortMAVsLongMA.csv"} <= written


def test_an_empty_out_keeps_no_persistent_output(md, tmp_path, monkeypatch):
    work_dir = tmp_path / "cwd"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    _, _, _, trades, _ = run_backtest(
        yaml.safe_load(STRATEGY), underlying="TEST", start_date="2020-01-01",
        end_date="2022-12-31", md=md, out="")

    assert len(trades) > 0
    assert list(work_dir.iterdir()) == []


def test_a_period_without_trades_evaluates_to_na(md):
    _, _, _, trades, stats = run_backtest(
        yaml.safe_load(STRATEGY), underlying="TEST", start_date="2020-01-01",
        end_date="2020-02-01", md=md, out="")

    assert len(trades) == 0
    assert stats['no_trades'] == 0
    assert "NA" in format_stats(stats)
    # a balance that sat in cash never moved and never drew down
    assert pd.isna(stats['sharpe'])
    assert pd.isna(stats['calmar'])
    assert stats['max_time_in_dd'] == 0


def evaluate(balances, trades=(), from_date=date(2020, 1, 1), to_date=date(2021, 12, 31),
             initial_position=100_000):
    """
        Runs the Evaluator over a hand written mark to market balance curve, one balance per
        consecutive calendar day, and over trades given as (shares, entry_price, exit_price).
        The balance is held entirely in cash, so 'pos' and 'C' do not enter the metrics.
    """
    positions = pd.DataFrame(
        {'cash': [float(b) for b in balances], 'pos': 0.0, 'C': 0.0},
        index=pd.date_range(from_date, periods=len(balances), freq='D'))
    trades = pd.DataFrame.from_records(
        [{'entry_date': pd.Timestamp(from_date), 'entry_price': entry, 'shares': shares,
          'exit_date': pd.Timestamp(to_date), 'exit_price': exit_price}
         for shares, entry, exit_price in trades],
        columns=['entry_date', 'entry_price', 'shares', 'exit_date', 'exit_price'])
    stats, _ = Evaluator().evaluate_trades(trades, positions, initial_position,
                                           from_date, to_date)
    return stats


def test_max_time_in_dd_is_the_longest_stretch_below_a_high_water_mark():
    # the Jan 2 peak of 110 stands under water from Jan 3 until it is recovered on Jan 8;
    # the shallower dip after the Jan 8 peak of 120 lasts a single day and must not win
    stats = evaluate([100, 110, 105, 100, 95, 108, 107, 120, 119, 121])

    assert stats['max_time_in_dd'] == 5


def test_a_strategy_still_under_water_is_measured_up_to_the_last_day():
    stats = evaluate([100, 120, 110, 105, 104])

    assert stats['max_time_in_dd'] == 3


def test_a_strategy_that_keeps_making_new_highs_spends_no_time_in_drawdown():
    stats = evaluate([100, 110, 120, 130])

    assert stats['max_time_in_dd'] == 0


def test_sharpe_annualizes_the_daily_mark_to_market_returns():
    # daily returns of 0.10, 0.10 and 0.00: mean 1/15, sample stdev 1/sqrt(300),
    # so sqrt(252) * mean / stdev = 18.33
    stats = evaluate([100, 110, 121, 121])

    assert stats['sharpe'] == 18.33


def test_calmar_is_the_cagr_per_unit_of_the_deepest_drawdown():
    # one trade turning 100,000 into 121,000 over two years is a CAGR of 0.10, and the
    # balance curve gives back 25,000 of its 125,000 peak for a max drawdown of 0.20
    stats = evaluate([100_000, 125_000, 100_000, 121_000], trades=[(1000, 100, 121)])

    assert stats['cagr'] == 0.10
    assert stats['max_dd_pcnt_unrealized'] == 0.20
    assert stats['calmar'] == 0.50


def test_stats_are_formatted_per_the_reporting_conventions():
    # the metrics are a mixed type Series, counts stay ints
    stats = pd.Series({'total_pnl': 48534.56, 'cagr': 0.0812, 'no_trades': 3, 'r': float('nan')},
                      dtype=object)

    lines = [line.split() for line in format_stats(stats).splitlines()]

    assert lines == [['total_pnl', '48,535'], ['cagr', '0.08'], ['no_trades', '3'], ['r', 'NA']]


def test_main_runs_a_backtest_from_the_command_line(strategy_file, tmp_path, capsys):
    out = tmp_path / "cli"

    rc = main(["--start_date", "2020-01-01", "--end_date", "2025-05-01",
               "-md", "md", "--underlying", "QQQ",
               "--strategy", strategy_file, "--out", str(out)])

    assert rc == 0
    captured = capsys.readouterr().out
    assert "Strategy   : ShortMAVsLongMA" in captured
    assert "no_trades" in captured
    assert out.is_dir()


def test_main_reports_a_bad_strategy_file_without_a_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("strategy:\n  entry_signal:\n    name: NoSuchSignal\n")

    rc = main(["--start_date", "2020-01-01", "-md", "md", "--underlying", "QQQ",
               "--strategy", str(bad)])

    assert rc == 2
    assert "Unknown entry signal" in capsys.readouterr().err
