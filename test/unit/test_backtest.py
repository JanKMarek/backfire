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
    PositionManagement,
)
from backfire.signals import ShortMAAboveLongMA, ShortMABelowLongMA

STRATEGY = """
strategy:
  name: ShortMAVsLongMA
  entry_signal:
    name: ShortMAAboveLongMA
    short_MA: 50
    long_MA: 200
  exit_signal:
    name: ShortMABelowLongMA
    short_MA: 50
    long_MA: 200
  risk_management:
    name: BasicRiskManagement
    stop_loss: 0.07
    take_profit: 0.25
    trailing_stop_period: null
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
    assert isinstance(s.exit_signal, ShortMABelowLongMA)
    assert isinstance(s.risk_management, BasicRiskManagement)
    assert (s.risk_management.stop_loss, s.risk_management.take_profit) == (0.07, 0.25)
    assert s.risk_management.trailing_stop_period is None
    assert isinstance(s.position_management, PositionManagement)
    assert s.position_management.initial_position == 100000


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
