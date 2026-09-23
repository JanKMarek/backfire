"""
    The signal verification report end to end on a short synthetic series: a decline, one
    Follow Through Day and a drift up afterwards, with a ground truth file naming the day
    before it and another date nothing fires near.
"""
from datetime import date

import pandas as pd
import pytest
import yaml

from backfire.ftd_signal_verification_report import (
    caption,
    fmt_date,
    fmt_pct,
    gallery_entries,
    load_signal_conf,
    main,
    panel_window,
    parse_args,
    run_report,
    scorecard_html,
    split_signal_overrides,
)

# day 3 is day 0, day 4 is day 1 and day 7 is the Follow Through Day - see test_signals.py -
# followed by a drift up, so the forward horizons have somewhere to go
CLOSES = [100.0, 98.0, 96.0, 93.0, 94.0, 94.5, 94.2, 96.2] + [97.0 + i for i in range(30)]
VOLUMES = [100.0] * 7 + [200.0] + [100.0] * 30
HORIZONS = (2, 5, 10)

SIGNAL = {'signal': {'name': 'FTDSignal', 'index': None, 'min_decline': 0.08,
                     'min_peak_age_days': 3, 'day0_window': 3, 'ftd_min_gain': 0.02,
                     'ftd_min_days': 4, 'ftd_max_days': 6}}


# the series runs on business days from 2020-01-01: the Follow Through Day is bar 7 and the
# ground truth date the report should find nothing near is bar 30
DAYS = list(pd.bdate_range("2020-01-01", periods=len(CLOSES)).date)
FTD_DAY, EMPTY_DAY = DAYS[7], DAYS[30]


@pytest.fixture(scope="module")
def md(tmp_path_factory):
    """ A market data directory holding the synthetic series as TEST.csv. """
    frame = pd.DataFrame({'Open': [c - 0.5 for c in CLOSES], 'High': [c + 1 for c in CLOSES],
                          'Low': [c - 1 for c in CLOSES], 'Close': CLOSES, 'Volume': VOLUMES},
                         index=pd.Index(DAYS, name='Date'))
    folder = tmp_path_factory.mktemp("md")
    frame.to_csv(folder / "TEST.csv")
    return str(folder)


@pytest.fixture(scope="module")
def turnarounds(tmp_path_factory):
    """ Turnaround points: one the day before the Follow Through Day, one nowhere near it. """
    path = tmp_path_factory.mktemp("truth") / "turns.csv"
    path.write_text("# the turnaround points\n\nDate,Notes\n"
                    f"{DAYS[6]},the one the rules should find\n"
                    f"{EMPTY_DAY},a date with no firing near it\n")
    return str(path)


@pytest.fixture(scope="module")
def references(tmp_path_factory):
    """ The same two dates as IBD calls: a reference on the day, and a candidate. """
    path = tmp_path_factory.mktemp("refs") / "refs.yaml"
    path.write_text(yaml.safe_dump({'follow_through_days': [
        {'date': FTD_DAY, 'episode': 'The turn', 'day1': DAYS[4], 'kind': 'reference',
         'confidence': None, 'notes': 'the one the rules should find'},
        {'date': EMPTY_DAY, 'episode': 'Nothing here', 'day1': None, 'kind': 'candidate',
         'confidence': 'low', 'notes': 'not an expectation'},
    ]}, sort_keys=False))
    return str(path)


@pytest.fixture(scope="module")
def report(tmp_path_factory, md, turnarounds):
    """ One full run; the report takes a few seconds to write, so it is built once. """
    return run_report(SIGNAL, underlying="TEST", start_date="2020-01-01", md=md,
                      out=str(tmp_path_factory.mktemp("out")), ground_truth=turnarounds,
                      early=1, late=3, horizons=HORIZONS, grid={'ftd_min_gain': [0.02, 0.05]})


# --- the command line ------------------------------------------------------------------

def test_the_signal_overrides_are_split_out_of_the_command_line():
    rest, overrides = split_signal_overrides(
        ["--underlying", "QQQ", "--signal", "s.yaml", "--signal.ftd_min_gain=0.015",
         "--signal.name=FTDSignal"])

    assert rest == ["--underlying", "QQQ", "--signal", "s.yaml"]
    assert overrides == ["signal.ftd_min_gain=0.015", "signal.name=FTDSignal"]


def test_a_signal_override_without_a_value_is_rejected():
    with pytest.raises(ValueError, match="--signal.<name>=<value>"):
        split_signal_overrides(["--signal.ftd_min_gain"])


def test_the_overrides_are_applied_on_top_of_the_signal_file(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(SIGNAL))

    conf = load_signal_conf(str(path), ["signal.ftd_min_gain=0.05"])

    assert conf['signal']['ftd_min_gain'] == 0.05
    assert conf['signal']['min_decline'] == 0.08        # untouched


def test_the_overrides_alone_can_define_the_signal():
    conf = load_signal_conf(None, ["signal.name=FTDSignal", "signal.index=null"])

    assert conf['signal'] == {'name': 'FTDSignal', 'index': None}


def test_a_run_without_any_signal_is_rejected():
    with pytest.raises(ValueError, match="No signal was defined"):
        load_signal_conf(None, [])


def test_parse_args_keeps_the_overrides_next_to_the_arguments():
    args = parse_args(["--underlying", "QQQ", "--start_date", "1999-03-10",
                       "--out", "out/x", "--signal.day0_window=10"])

    assert (args.underlying, args.start_date, args.out) == ("QQQ", "1999-03-10", "out/x")
    assert args.overrides == ["signal.day0_window=10"]
    # the verification parameters default to what docs/SIGNALS.md gives
    assert (args.ground_truth, args.early_days, args.late_days) == ("turnarounds", 1, 3)


def test_parse_args_takes_the_ground_truth_and_the_tolerance():
    args = parse_args(["--underlying", "QQQ", "--start_date", "1999-03-10", "--out", "out/x",
                       "--ground_truth", "ibd", "--early_days", "0", "--late_days", "5"])

    assert (args.ground_truth, args.early_days, args.late_days) == ("ibd", 0, 5)


def test_parse_args_rejects_a_negative_tolerance():
    with pytest.raises(SystemExit):
        parse_args(["--underlying", "QQQ", "--start_date", "1999-03-10", "--out", "out/x",
                    "--late_days", "-1"])


def test_parse_args_rejects_a_date_that_is_not_a_date():
    with pytest.raises(SystemExit):
        parse_args(["--underlying", "QQQ", "--start_date", "March", "--out", "out/x"])


# --- the report ------------------------------------------------------------------------

def test_the_run_writes_the_report_and_its_three_csv_files(report):
    written = report['paths']

    assert set(written) == {'report', 'scorecard', 'episodes', 'signal_values'}
    assert written['report'].endswith("ftd_signal_verification_report.html")
    for path in written.values():
        assert open(path, encoding="utf-8").read()
    assert pd.read_csv(written['episodes']).outcome.tolist() == ['FTD']
    scorecard = pd.read_csv(written['scorecard'])
    assert scorecard.hit.tolist() == [True, False]
    assert scorecard.notes.tolist() == ['the one the rules should find',
                                        'a date with no firing near it']


def test_the_report_is_self_contained_html(report):
    page = open(report['paths']['report'], encoding="utf-8").read()

    assert page.startswith("<!DOCTYPE html>")
    assert page.rstrip().endswith("</html>")
    assert "Plotly.newPlot" in page
    # plotly is inlined rather than fetched, so the report works offline
    assert "<script src=" not in page and "<link " not in page


def test_the_header_carries_the_statistics(report):
    page = open(report['paths']['report'], encoding="utf-8").read()

    assert ">2</b><span>ground truth dates in the data" in page
    assert ">1</b><span>positives (firings)" in page
    assert f">{len(CLOSES)}</b><span>trading days" in page
    assert ">1</b><span>true positives" in page
    assert ">0</b><span>false positives" in page
    assert ">100.0%</b><span>precision" in page
    assert ">50.0%</b><span>recall" in page
    assert ">0.67</b><span>F1" in page


def test_the_parameters_spell_out_the_run_and_the_test(report):
    page = open(report['paths']['report'], encoding="utf-8").read()

    assert "<td>ftd_min_gain</td><td>0.02</td>" in page
    assert "<td>underlying</td><td>TEST</td>" in page
    assert "from 1 trading day before it to 3 after it" in page


def test_the_scorecard_has_a_row_per_ground_truth_date(report):
    page = open(report['paths']['report'], encoding="utf-8").read()

    # the matched date links to its panel and names the firing, its gap and its rally day
    assert (f'<a href="#truth-{DAYS[6]}">{fmt_date(DAYS[6])}</a></td>'
            f'<td>{fmt_date(FTD_DAY)} (+1d)</td><td>true positive</td><td>4</td>') in page
    # the missed one carries the reason instead
    assert (f'<a href="#truth-{EMPTY_DAY}">{fmt_date(EMPTY_DAY)}</a></td><td></td>'
            f'<td>miss</td><td></td><td></td><td></td>'
            f'<td>still in the uptrend of the {FTD_DAY} follow through day</td>') in page
    # there is no separate false positives table any more
    assert "<h3>False positives" not in page


def test_the_scorecard_interleaves_the_false_positives_in_date_order(report):
    false_positives = [DAYS[2], DAYS[20], DAYS[35]]
    table = scorecard_html(report['scorecard'], false_positives, report['episodes'])

    rows = [f'href="#fp-{DAYS[2]}"', f'href="#truth-{DAYS[6]}"', f'href="#fp-{DAYS[20]}"',
            f'href="#truth-{EMPTY_DAY}"', f'href="#fp-{DAYS[35]}"']
    positions = [table.index(row) for row in rows]
    assert positions == sorted(positions)
    assert table.count("<td>false positive</td>") == 3


def test_the_page_has_the_analysis_sections(report):
    page = open(report['paths']['report'], encoding="utf-8").read()

    assert "<h2>What followed a firing</h2>" in page
    assert "naive FTD days" in page
    assert "<h2>The rally attempts</h2>" in page
    assert 'id="rally-day"' in page
    assert "<h2>Sensitivity to the parameters</h2>" in page
    assert "<td>ftd_min_gain=0.05</td><td>0</td><td>0 / 2</td>" in page


def test_the_paths_are_split_into_successful_and_failed_follow_through_days(report):
    page = open(report['paths']['report'], encoding="utf-8").read()
    stats = report['path_stats']

    # the one uptrend is still open when the data ends: it counts under all FTDs only
    assert stats['count'].tolist() == [1, 0, 0]
    # the paths run 10 days here, so only the 5 and 10 day medians are shown
    assert stats.columns.tolist() == ['count', 'median r5', 'median r10']
    assert "<h3>Along the path, successful against failed</h3>" in page
    assert "<th>median @5d</th><th>median @10d</th></tr>" in page
    assert "<td>failed FTDs</td><td>0</td><td>-</td><td>-</td>" in page


def test_the_gallery_has_a_panel_per_ground_truth_date(report):
    page = open(report['paths']['report'], encoding="utf-8").read()

    assert f'id="truth-{DAYS[6]}"' in page          # the matched date
    assert f'id="truth-{EMPTY_DAY}"' in page        # the date with nothing near it
    assert 'id="fp-' not in page                    # no false positives to show


def test_the_report_reproduces_what_the_analysis_says(report):
    assert int(report['signal_values'].es.sum()) == 1
    assert report['scorecard'].hit.tolist() == [True, False]
    assert report['stats']['true_positives'] == 1
    assert report['false_positives'] == []
    assert report['paths_after'].columns.tolist() == list(range(1, max(HORIZONS) + 1))
    assert report['sensitivity'].setting.tolist() == ['base', 'ftd_min_gain=0.05']
    assert report['sensitivity'].hits.tolist() == [1, 0]


def test_the_ibd_calls_are_the_other_ground_truth(md, references):
    rv = run_report(SIGNAL, underlying="TEST", start_date="2020-01-01", md=md, out="",
                    ground_truth=references, horizons=HORIZONS, grid={})

    # only the reference counts; the candidate is not an expectation
    assert rv['scorecard'].date.tolist() == [FTD_DAY]
    assert rv['scorecard'].hit.tolist() == [True]
    assert rv['stats']['recall'] == 1.0


def test_a_firing_no_ground_truth_date_claims_is_a_false_positive(md, tmp_path):
    truth = tmp_path / "turns.csv"
    truth.write_text(f"Date,Notes\n{EMPTY_DAY},nothing fires here\n")

    rv = run_report(SIGNAL, underlying="TEST", start_date="2020-01-01", md=md, out="",
                    ground_truth=str(truth), horizons=HORIZONS, grid={})

    assert rv['false_positives'] == [FTD_DAY]
    assert (rv['stats']['precision'], rv['stats']['recall']) == (0.0, 0.0)
    assert f'id="fp-{FTD_DAY}"' in rv['html']
    assert f'href="#fp-{FTD_DAY}"' in rv['html']


def test_a_signal_that_does_not_explain_itself_is_refused(tmp_path, md, turnarounds):
    with pytest.raises(ValueError, match="does not record the day by day diagnostics"):
        run_report({'signal': {'name': 'AlwaysOnSignal'}}, underlying="TEST",
                   start_date="2020-01-01", md=md, out="", ground_truth=turnarounds)


def test_a_run_with_no_output_folder_writes_nothing_and_still_analyses(md, turnarounds):
    rv = run_report(SIGNAL, underlying="TEST", start_date="2020-01-01", md=md, out="",
                    ground_truth=turnarounds, horizons=HORIZONS, grid={})

    assert rv['paths'] == {}
    assert len(rv['episodes']) == 1


def test_main_reports_the_run_and_returns_zero(tmp_path, md, turnarounds, capsys):
    rc = main(["--underlying", "TEST", "--start_date", "2020-01-01", "-md", md,
               "--out", str(tmp_path / "out"), "--ground_truth", turnarounds,
               "--signal.name=FTDSignal", "--signal.index=null",
               "--signal.min_peak_age_days=3", "--signal.day0_window=3",
               "--signal.ftd_min_gain=0.02", "--signal.ftd_min_days=4",
               "--signal.ftd_max_days=6"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Positives    : 1 (1 true, 0 false)" in out
    assert "Precision    : 100.0%" in out
    assert "Recall       : 50.0%" in out


def test_main_reports_a_failure_instead_of_raising(tmp_path, capsys):
    rc = main(["--underlying", "NOPE", "--start_date", "2020-01-01",
               "-md", str(tmp_path), "--out", str(tmp_path / "out"),
               "--signal.name=NoSuchSignal"])

    assert rc == 2
    assert "report failed" in capsys.readouterr().err


# --- the panels ------------------------------------------------------------------------

def entries_of(report):
    return gallery_entries(report['scorecard'], report['episodes'],
                           list(report['signal_values'].index[report['signal_values'].es]))


def test_the_gallery_lists_the_ground_truth_dates_in_scorecard_order(report):
    entries = entries_of(report)

    assert [e['group'] for e in entries] == ['truth', 'truth']
    assert [e['date'] for e in entries] == [DAYS[6], EMPTY_DAY]
    assert entries[0]['episode'] is not None        # the date was matched
    assert entries[1]['episode'] is None


def test_a_matched_panel_reads_as_a_verdict(report):
    entries = entries_of(report)
    outcomes = pd.DataFrame(index=[])

    rv = caption(entries[0], outcomes)

    assert "ground truth Jan 9 2020" in rv
    assert "fired Jan 10 2020 (+1d)" in rv
    assert "day 4 of the attempt" in rv


def test_a_missed_panel_says_why_nothing_fired(report):
    entries = entries_of(report)

    rv = caption(entries[1], pd.DataFrame(index=[]))

    assert "not fired - still in the uptrend of the 2020-01-10 follow through day" in rv


def test_a_panel_covers_its_episode_with_context_on_each_side(report):
    bars = report['ohlcv'].index
    bar_of = {day: i for i, day in enumerate(bars)}
    entries = entries_of(report)

    first, last = panel_window(entries[0], bar_of, bars)

    # the series is shorter than the context on either side, so the panel is the whole of it
    assert (first, last) == (0, len(bars) - 1)


# --- formatting ------------------------------------------------------------------------

def test_dates_and_percentages_are_formatted_for_reading():
    assert fmt_date(date(2020, 4, 2)) == "Apr 2 2020"
    assert fmt_date(None) == "-"
    assert fmt_pct(0.1834) == "+18.3%"
    assert fmt_pct(-0.05, digits=0) == "-5%"
    assert fmt_pct(0.5, sign=False) == "50.0%"
    assert fmt_pct(float('nan')) == "-"
