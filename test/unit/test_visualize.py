import numpy as np
import pandas as pd
import pytest

from backfire import visualize
from backfire.visualize import (
    build_figure,
    drawdown_episodes,
    figure_title,
    find_run_files,
    highlight_for_cell,
    key_statistics,
    load_positions,
    load_run,
    load_stats,
    load_trades,
    main,
    period_bounds,
    returns_pivot,
    returns_style,
    signal_blocks,
    signal_names,
    stat_value,
    table_records,
)


def _write_positions(path, dates, balance, es=None, xs=None, memo=None,
                     with_date_header=True):
    """
        Writes a minimal pos_*.csv, shaped like the one SignalDrivenStrategy.backtest
        writes, with the columns visualize.py reads plus enough of the rest to look
        like the real file.
    """
    es = pd.Series(0.0, index=dates) if es is None else es
    xs = pd.Series(0.0, index=dates) if xs is None else xs
    memo = pd.Series(np.nan, index=dates, dtype=object) if memo is None else memo
    close = pd.Series(balance, index=dates) / 1000.0

    df = pd.DataFrame({
        'O': close, 'H': close + 1, 'L': close - 1, 'C': close, 'V': 1_000_000.0,
        'es': es, 'es_id': np.nan, 'xs': xs, 'pos': 0, 'cash': balance,
        'action': np.nan, 'delta_shares': 0, 'memo': memo,
        'buy_price': np.nan, 'trailing_stop': np.nan,
        'balance': balance, 'unrealized_CumMax': 0.0, 'unrealized_dd': 0.0,
        'unrealized_dd_pcnt': 0.0,
    }, index=pd.Index(dates, name='Date' if with_date_header else None))
    df.to_csv(path, float_format='%.2f')


@pytest.fixture
def run_folder(tmp_path):
    """
        A three year run written the way Strategy.backtest writes one: a balance curve
        that rises, dips and recovers, one long entry signal regime, one single day
        entry signal, two trades, a stats file written before sharpe/calmar/
        max_time_in_dd existed, and a subdirectory holding a second run that must be
        ignored.
    """
    folder = tmp_path / "run"
    folder.mkdir()
    # ends mid-year so the trailing months of the last year are genuinely untraded
    dates = pd.bdate_range("2020-01-01", "2022-06-15")
    n = len(dates)

    balance = 100_000 + np.linspace(0, 30_000, n)
    lo, hi = n // 3, 2 * n // 3
    balance[lo:hi] -= np.linspace(0, 15_000, hi - lo)

    es = pd.Series(0.0, index=dates)
    es.iloc[10:200] = 100.0            # one long entry regime
    es.iloc[500] = 100.0               # one single day entry
    xs = pd.Series(0.0, index=dates)
    xs.iloc[199:210] = 100.0

    memo = pd.Series(np.nan, index=dates, dtype=object)
    memo.iloc[11] = "bought:100 shares;EntrySignal"
    memo.iloc[200] = "sold:-100 shares;ExitSignal"
    memo.iloc[501] = "bought:50 shares;EntrySignal"
    memo.iloc[520] = "sold:-50 shares;sl"

    _write_positions(folder / "pos_TEST_Strategy.csv", dates, balance, es=es, xs=xs, memo=memo)

    stats = pd.Series({
        'no_trades': 2, 'no_winning_trades': 1, 'no_losing_trades': 1,
        'total_pnl': 300.0, 'rtn': 1.15, 'time_span': '1095 days, 0:00:00', 'cagr': 0.05,
        'max_dd_pcnt_realized': 0.02, 'max_dd_pcnt_unrealized': 0.13,
        # sharpe, calmar and max_time_in_dd deliberately omitted: reproduces a run
        # written before those metrics existed
    }, name='stats')
    stats.to_csv(folder / "stats_TEST_Strategy.csv", header=True, index=True)

    trades = pd.DataFrame([
        {'ticker': 'TEST', 'entry_date': dates[11], 'entry_price': balance[11] / 1000,
         'shares': 100, 'exit_date': dates[200], 'exit_price': balance[200] / 1000,
         'memo': 'bought:100 shares;EntrySignal / sold:-100 shares;ExitSignal',
         'pnl': 500.0, 'pnl_pcnt': 0.05, 'hp': 189},
        {'ticker': 'TEST', 'entry_date': dates[501], 'entry_price': balance[501] / 1000,
         'shares': 50, 'exit_date': dates[520], 'exit_price': balance[520] / 1000,
         'memo': 'bought:50 shares;EntrySignal / sold:-50 shares;sl',
         'pnl': -200.0, 'pnl_pcnt': -0.02, 'hp': 19},
    ])
    trades.to_csv(folder / "trades_TEST_Strategy.csv", header=True, index=True)

    nested = folder / "nested"
    nested.mkdir()
    _write_positions(nested / "pos_OTHER_Ignored.csv", dates[:50], balance[:50])

    return str(folder)


# ------------------------------------------------------------------------------
# Discovery and loading
# ------------------------------------------------------------------------------

def test_the_ticker_and_strategy_are_read_off_the_positions_file_name(run_folder):
    files = find_run_files(run_folder)

    assert files['ticker'] == "TEST"
    assert files['strategy_name'] == "Strategy"


def test_underscores_inside_the_strategy_name_survive(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    dates = pd.bdate_range("2020-01-01", periods=10)
    _write_positions(folder / "pos_QQQ_Index_50dMAvs200dMA.csv", dates, [100_000.0] * 10)

    files = find_run_files(str(folder))

    assert (files['ticker'], files['strategy_name']) == ("QQQ", "Index_50dMAvs200dMA")


def test_subdirectories_of_the_run_folder_are_ignored(run_folder):
    files = find_run_files(run_folder)

    assert files['positions'].endswith("pos_TEST_Strategy.csv")


def test_a_folder_without_a_positions_file_is_reported(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match="pos_"):
        find_run_files(str(empty))


def test_a_folder_holding_more_than_one_run_names_them_all(tmp_path):
    folder = tmp_path / "many"
    folder.mkdir()
    dates = pd.bdate_range("2020-01-01", periods=10)
    _write_positions(folder / "pos_A_X.csv", dates, [100_000.0] * 10)
    _write_positions(folder / "pos_B_Y.csv", dates, [100_000.0] * 10)

    with pytest.raises(ValueError, match="pos_A_X.csv") as exc_info:
        find_run_files(str(folder))
    assert "pos_B_Y.csv" in str(exc_info.value)


@pytest.mark.parametrize("with_date_header", [True, False])
def test_positions_load_regardless_of_whether_the_date_header_survived(tmp_path, with_date_header):
    dates = pd.bdate_range("2020-01-01", periods=10)
    path = tmp_path / "pos.csv"
    _write_positions(path, dates, [100_000.0 + i for i in range(10)],
                     with_date_header=with_date_header)

    df = load_positions(str(path))

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == 'Date'
    assert list(df['balance']) == [100_000.0 + i for i in range(10)]


def test_a_stats_file_written_before_a_metric_existed_reports_it_as_missing(run_folder):
    files = find_run_files(run_folder)
    stats = load_stats(files['stats'])

    assert stat_value(stats, 'sharpe') is None
    assert stat_value(stats, 'cagr') == 0.05


def test_stats_are_empty_when_the_run_wrote_none():
    assert load_stats(None).empty


def test_trades_are_empty_with_the_expected_columns_when_the_run_wrote_none():
    trades = load_trades(None)

    assert trades.empty
    assert list(trades.columns) == ['ticker', 'entry_date', 'entry_price', 'shares',
                                    'exit_date', 'exit_price', 'memo', 'pnl', 'pnl_pcnt', 'hp']


def test_signal_names_are_recovered_from_the_trade_memos_skipping_risk_labels(run_folder):
    run = load_run(run_folder)

    # the second trade was stopped out ('sl'), so the exit signal name must come from
    # the first trade's memo, not from the risk management label
    assert run.entry_signal_name == "EntrySignal"
    assert run.exit_signal_name == "ExitSignal"


def test_signal_names_are_none_when_a_side_never_traded():
    dates = pd.bdate_range("2020-01-01", periods=10)
    positions = pd.DataFrame({'memo': [np.nan] * 10}, index=dates)

    assert signal_names(positions) == (None, None)


# ------------------------------------------------------------------------------
# Drawdowns
# ------------------------------------------------------------------------------

def test_the_three_largest_drawdowns_are_the_deepest_point_of_three_episodes():
    balance = pd.Series([100, 90, 100, 80, 100, 95, 100],
                        index=pd.date_range("2020-01-01", periods=7))

    episodes = drawdown_episodes(balance)

    assert [round(e['value'], 2) for e in episodes] == [-0.2, -0.1, -0.05]
    assert episodes[0]['date'] == pd.Timestamp("2020-01-04")


def test_one_slide_is_reported_once_however_long_it_lasts():
    balance = pd.Series([100] + list(range(99, 59, -1)),
                        index=pd.date_range("2020-01-01", periods=41))

    episodes = drawdown_episodes(balance)

    assert len(episodes) == 1


def test_a_balance_that_only_rises_has_no_drawdowns():
    balance = pd.Series([100, 110, 120, 130], index=pd.date_range("2020-01-01", periods=4))

    assert drawdown_episodes(balance) == []


def test_an_episode_still_under_water_ends_on_the_last_day():
    balance = pd.Series([100, 120, 110, 105, 104], index=pd.date_range("2020-01-01", periods=5))

    episodes = drawdown_episodes(balance)

    assert episodes[0]['end'] == balance.index[-1]


def test_only_the_requested_number_of_episodes_is_returned():
    balance = pd.Series([100, 90, 100, 80, 100, 70, 100, 95, 100],
                        index=pd.date_range("2020-01-01", periods=9))

    assert len(drawdown_episodes(balance, top=2)) == 2


# ------------------------------------------------------------------------------
# Header statistics
# ------------------------------------------------------------------------------

def test_total_return_and_cagr_come_from_the_stats_file_when_present():
    balance = pd.Series([100_000.0, 110_000.0], index=pd.date_range("2020-01-01", periods=2))
    run = visualize.Run("folder", "TEST", "Strategy",
                        pd.DataFrame({'balance': balance, 'memo': [np.nan, np.nan]},
                                    index=balance.index),
                        pd.Series({'rtn': 1.15, 'cagr': 0.05}), load_trades(None))

    stats = key_statistics(run)

    assert stats['total_return'] == pytest.approx(0.15)
    assert stats['cagr'] == 0.05


def test_total_return_and_cagr_are_computed_from_balance_when_the_stats_file_is_empty():
    dates = pd.date_range("2020-01-01", periods=366)
    balance = pd.Series([100_000.0] * 365 + [121_000.0], index=dates)
    run = visualize.Run("folder", "TEST", "Strategy",
                        pd.DataFrame({'balance': balance,
                                     'memo': [np.nan] * len(dates)}, index=dates),
                        pd.Series(dtype=object), load_trades(None))

    stats = key_statistics(run)

    assert stats['total_return'] == pytest.approx(0.21)
    assert stats['cagr'] == pytest.approx(0.21, abs=1e-2)
    assert stats['sharpe'] is None


def test_key_statistics_does_not_raise_for_a_run_folder(run_folder):
    run = load_run(run_folder)

    stats = key_statistics(run)

    assert stats['no_trades'] == 2
    assert len(stats['drawdowns']) > 0


# ------------------------------------------------------------------------------
# Monthly / annual returns table
# ------------------------------------------------------------------------------

def test_monthly_returns_compound_to_the_annual_return(run_folder):
    run = load_run(run_folder)
    pivot = returns_pivot(run.balance)

    year_columns = [c for c in pivot.columns if c != 'Month']
    months = pivot[pivot['Month'] != 'Annual']
    annual = pivot[pivot['Month'] == 'Annual'].iloc[0]

    for year in year_columns:
        compounded = (1 + months[year].dropna()).prod() - 1
        if pd.notna(annual[year]):
            assert compounded == pytest.approx(annual[year], abs=1e-9)


def test_the_returns_table_has_twelve_month_rows_and_an_annual_row_last(run_folder):
    run = load_run(run_folder)
    pivot = returns_pivot(run.balance)

    assert pivot['Month'].tolist() == [
        'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
        'September', 'October', 'November', 'December', 'Annual']
    assert all(isinstance(c, str) for c in pivot.columns if c != 'Month')


def test_months_outside_the_trading_period_are_empty_cells(run_folder):
    run = load_run(run_folder)
    pivot = returns_pivot(run.balance)
    records = table_records(pivot)

    last_year = str(run.end.year)
    december_row = next(r for r in records if r['Month'] == 'December')
    assert december_row[last_year] is None


# ------------------------------------------------------------------------------
# Signal blocks
# ------------------------------------------------------------------------------

def test_signal_blocks_collapse_contiguous_signal_days_into_ranges():
    dates = pd.date_range("2020-01-01", periods=60)
    signal = pd.Series(0.0, index=dates)
    signal.iloc[0:40] = 10.0
    signal.iloc[50:55] = 10.0

    blocks = signal_blocks(signal)

    assert len(blocks) == 2
    assert blocks[0] == (dates[0], dates[39])
    assert blocks[1] == (dates[50], dates[54])


def test_a_single_day_signal_is_widened_so_that_it_stays_visible():
    dates = pd.date_range("2020-01-01", periods=10)
    signal = pd.Series(0.0, index=dates)
    signal.iloc[5] = 10.0

    (start, end), = signal_blocks(signal, min_days=3)

    assert (end - start).days + 1 >= 3


def test_a_long_block_is_not_widened():
    dates = pd.date_range("2020-01-01", periods=10)
    signal = pd.Series(0.0, index=dates)
    signal.iloc[0:8] = 10.0

    (start, end), = signal_blocks(signal, min_days=3)

    assert (start, end) == (dates[0], dates[7])


def test_signal_blocks_are_empty_when_the_signal_never_fires():
    dates = pd.date_range("2020-01-01", periods=10)
    signal = pd.Series(0.0, index=dates)

    assert signal_blocks(signal) == []


# ------------------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------------------

def test_the_figure_plots_the_balance_the_underlying_and_both_signal_bands(run_folder):
    run = load_run(run_folder)

    fig = build_figure(run)

    names = [trace.name for trace in fig.data]
    assert "EntrySignal on" in names
    assert "ExitSignal on" in names
    assert fig.layout.yaxis.type == 'log'
    assert fig.layout.yaxis2.type == 'log'
    assert fig.layout.yaxis3.visible is False
    assert fig.layout.xaxis.rangeslider.visible is False
    band_traces = [t for t in fig.data if t.name in ("EntrySignal on", "ExitSignal on")]
    assert all(t.yaxis == 'y3' for t in band_traces)


def test_the_figure_marks_the_executed_trades_on_the_price_axis(run_folder):
    run = load_run(run_folder)

    fig = build_figure(run)

    entry_trace = next(t for t in fig.data if t.name == 'entry')
    exit_trace = next(t for t in fig.data if t.name == 'exit')
    assert entry_trace.marker.symbol == 'triangle-up'
    assert exit_trace.marker.symbol == 'triangle-down'
    assert len(entry_trace.x) == 2


def test_the_hover_label_shows_only_near_the_underlying(run_folder):
    run = load_run(run_folder)

    fig = build_figure(run)

    assert fig.layout.hovermode == 'closest'
    price_hover = next(t for t in fig.data if t.name == 'price hover')
    assert price_hover.yaxis == 'y2'
    assert list(price_hover.y) == list(run.positions['C'])
    assert price_hover.marker.opacity == 0
    ohlc = next(t for t in fig.data if t.type == 'ohlc')
    balance = next(t for t in fig.data if t.name.endswith('portfolio value'))
    assert ohlc.hoverinfo == 'skip'
    assert balance.hoverinfo == 'skip'


def test_trade_markers_are_absent_when_there_are_no_trades(run_folder):
    run = load_run(run_folder)
    run.trades = load_trades(None)

    fig = build_figure(run)

    assert not any(t.name in ('entry', 'exit') for t in fig.data)


def test_the_figure_title_reports_the_strategy_the_period_and_the_key_statistics(run_folder):
    run = load_run(run_folder)
    stats = key_statistics(run)

    title = figure_title(run, stats)

    assert run.strategy_name in title
    assert f"{run.start:%Y-%m-%d}" in title
    assert f"{run.end:%Y-%m-%d}" in title
    assert "Total return" in title
    assert "CAGR" in title
    assert "Top drawdowns" in title


def test_the_title_omits_the_drawdown_clause_for_a_curve_that_only_rises():
    dates = pd.date_range("2020-01-01", periods=4)
    balance = pd.Series([100.0, 110.0, 120.0, 130.0], index=dates)
    run = visualize.Run("folder", "TEST", "Strategy",
                        pd.DataFrame({'balance': balance, 'memo': [np.nan] * 4}, index=dates),
                        pd.Series(dtype=object), load_trades(None))

    title = figure_title(run, key_statistics(run))

    assert "Top drawdowns" not in title


# ------------------------------------------------------------------------------
# Click to highlight
# ------------------------------------------------------------------------------

def test_clicking_a_month_cell_marks_the_first_and_the_last_day_of_that_month():
    assert period_bounds('February', '2024') == (pd.Timestamp('2024-02-01'),
                                                  pd.Timestamp('2024-02-29'))


def test_clicking_an_annual_cell_marks_the_whole_year():
    assert period_bounds('Annual', '2021') == (pd.Timestamp('2021-01-01'),
                                                pd.Timestamp('2021-12-31'))


def test_clicking_the_month_label_column_names_no_period():
    assert period_bounds('Month', 'Month') is None


def test_clicking_the_month_label_column_clears_the_highlight():
    rows = [{'Month': 'January', '2020': 0.1}]
    active_cell = {'row': 0, 'column_id': 'Month'}

    shapes, styles = highlight_for_cell(active_cell, rows, ['2020'])

    assert shapes == []
    assert not any(s.get('backgroundColor') == 'yellow' for s in styles)


def test_no_active_cell_leaves_the_chart_unmarked():
    shapes, styles = highlight_for_cell(None, [], ['2020'])

    assert shapes == []
    assert styles == returns_style(['2020'])


def test_clicking_a_cell_marks_it_and_installs_two_vertical_markers():
    rows = [{'Month': 'January', '2020': 0.1}, {'Month': 'February', '2020': -0.1}]
    active_cell = {'row': 1, 'column_id': '2020'}

    shapes, styles = highlight_for_cell(active_cell, rows, ['2020'])

    assert len(shapes) == 2
    assert all(s['yref'] == 'paper' for s in shapes)
    yellow = [s for s in styles if s.get('backgroundColor') == 'yellow']
    assert yellow == [{'if': {'row_index': 1, 'column_id': '2020'},
                       'backgroundColor': 'yellow', 'color': 'black'}]


def test_the_returns_table_colours_gains_green_and_losses_red():
    style = returns_style(['2020'])

    assert {'if': {'filter_query': '{2020} > 0.05', 'column_id': '2020'},
           'backgroundColor': 'lightgreen', 'color': 'black'} in style
    assert {'if': {'filter_query': '{2020} < -0.05', 'column_id': '2020'},
           'backgroundColor': 'lightcoral', 'color': 'black'} in style


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def test_main_serves_the_dashboard_on_the_requested_host_and_port(run_folder, monkeypatch, capsys):
    served = {}
    monkeypatch.setattr(visualize, "serve",
                        lambda app, host, port, debug: served.update(
                            app=app, host=host, port=port, debug=debug))

    rc = main(["-out", run_folder, "--port", "9000"])

    assert rc == 0
    assert served['port'] == 9000
    captured = capsys.readouterr().out
    assert "Strategy   : Strategy" in captured
    assert "Serving    : http://127.0.0.1:9000/" in captured


@pytest.mark.parametrize("flag", ["-out", "--out"])
def test_both_out_spellings_name_the_run_folder(run_folder, monkeypatch, flag):
    monkeypatch.setattr(visualize, "serve", lambda app, host, port, debug: None)

    rc = main([flag, run_folder])

    assert rc == 0


def test_main_reports_a_missing_run_folder_without_a_traceback(tmp_path, monkeypatch, capsys):
    served = []
    monkeypatch.setattr(visualize, "serve", lambda *a, **k: served.append(True))

    rc = main(["-out", str(tmp_path / "nope")])

    assert rc == 2
    assert "visualize failed" in capsys.readouterr().err
    assert served == []


def test_main_rejects_a_port_outside_the_valid_range(run_folder):
    with pytest.raises(SystemExit):
        main(["-out", run_folder, "--port", "99999"])
