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
from backfire.base import BasicRiskManagement, Environment, PositionManagement
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
