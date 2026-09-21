"""
    The signal analysis report end to end on a short synthetic series: a decline, one Follow
    Through Day and a drift up afterwards, with a reference file naming that day.
"""
from datetime import date

import pandas as pd
import pytest
import yaml

from backfire.report_signal import (
    caption,
    fmt_date,
    fmt_pct,
    gallery_entries,
    load_signal_conf,
    main,
    panel_window,
    parse_args,
    run_report,
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
# candidate the report should find nothing near is bar 30
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
def references(tmp_path_factory):
    """ One reference on the Follow Through Day and one candidate nowhere near it. """
    path = tmp_path_factory.mktemp("refs") / "refs.yaml"
    path.write_text(yaml.safe_dump({'follow_through_days': [
        {'date': FTD_DAY, 'episode': 'The turn', 'day1': DAYS[4], 'kind': 'reference',
         'confidence': None, 'notes': 'the one the rules should find'},
        {'date': EMPTY_DAY, 'episode': 'Nothing here', 'day1': None, 'kind': 'candidate',
         'confidence': 'low', 'notes': 'a date with no firing near it'},
    ]}, sort_keys=False))
    return str(path)


@pytest.fixture(scope="module")
def report(tmp_path_factory, md, references):
    """ One full run; the report takes a few seconds to write, so it is built once. """
    return run_report(SIGNAL, underlying="TEST", start_date="2020-01-01", md=md,
                      out=str(tmp_path_factory.mktemp("out")), references=references,
                      grid={'ftd_min_gain': [0.02, 0.05]}, tolerance=3, horizons=HORIZONS)


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


def test_parse_args_rejects_a_date_that_is_not_a_date():
    with pytest.raises(SystemExit):
        parse_args(["--underlying", "QQQ", "--start_date", "March", "--out", "out/x"])


# --- the report ------------------------------------------------------------------------

def test_the_run_writes_the_report_and_its_three_csv_files(report):
    written = report['paths']

    assert set(written) == {'report', 'episodes', 'references', 'signal_values'}
    for path in written.values():
        assert open(path, encoding="utf-8").read()
    assert pd.read_csv(written['episodes']).outcome.tolist() == ['FTD']
    assert pd.read_csv(written['references']).episode.tolist() == ['The turn', 'Nothing here']


def test_the_report_is_self_contained_html(report):
    page = open(report['paths']['report'], encoding="utf-8").read()

    assert page.startswith("<!DOCTYPE html>")
    assert page.rstrip().endswith("</html>")
    assert "Plotly.newPlot" in page
    # plotly is inlined rather than fetched, so the report works offline
    assert "<script src=" not in page and "<link " not in page


def test_the_header_carries_the_firings_and_the_reference_score(report):
    page = open(report['paths']['report'], encoding="utf-8").read()

    assert "1 / 1</b><span>reference dates hit" in page
    assert ">1</b><span>firings" in page


def test_the_gallery_has_a_panel_per_reference_and_candidate(report):
    page = open(report['paths']['report'], encoding="utf-8").read()

    assert f'id="ref-{FTD_DAY}"' in page            # the Follow Through Day
    assert f'id="cand-{EMPTY_DAY}"' in page         # the candidate with nothing near it
    assert f'href="#ref-{FTD_DAY}"' in page         # the match table links to the panel


def test_the_report_reproduces_what_the_analysis_says(report):
    assert int(report['signal_values'].es.sum()) == 1
    assert report['references'].hit.tolist() == [True, False]
    assert report['sensitivity'].setting.tolist() == ['base', 'ftd_min_gain=0.05']


def test_a_signal_that_does_not_explain_itself_is_refused(tmp_path, md, references):
    with pytest.raises(ValueError, match="does not record the day by day diagnostics"):
        run_report({'signal': {'name': 'AlwaysOnSignal'}}, underlying="TEST",
                   start_date="2020-01-01", md=md, out="", references=references)


def test_a_run_with_no_output_folder_writes_nothing_and_still_analyses(md, references):
    rv = run_report(SIGNAL, underlying="TEST", start_date="2020-01-01", md=md, out="",
                    references=references, grid={}, tolerance=3, horizons=HORIZONS)

    assert rv['paths'] == {}
    assert len(rv['episodes']) == 1


def test_main_reports_the_run_and_returns_zero(tmp_path, md, references, capsys):
    rc = main(["--underlying", "TEST", "--start_date", "2020-01-01", "-md", md,
               "--out", str(tmp_path / "out"), "--references", references,
               "--signal.name=FTDSignal", "--signal.index=null",
               "--signal.min_peak_age_days=3", "--signal.day0_window=3",
               "--signal.ftd_min_gain=0.02", "--signal.ftd_min_days=4",
               "--signal.ftd_max_days=6"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Firings    : 1" in out
    assert "References : 1 of 1 hit" in out


def test_main_reports_a_failure_instead_of_raising(tmp_path, capsys):
    rc = main(["--underlying", "NOPE", "--start_date", "2020-01-01",
               "-md", str(tmp_path), "--out", str(tmp_path / "out"),
               "--signal.name=NoSuchSignal"])

    assert rc == 2
    assert "report failed" in capsys.readouterr().err


# --- the panels ------------------------------------------------------------------------

def test_the_gallery_lists_the_references_matched_first_then_the_candidates(report):
    entries = gallery_entries(report['references'], report['episodes'], tolerance=3)

    assert [e['group'] for e in entries] == ['reference', 'candidate']
    assert entries[0]['episode'] is not None        # the reference was matched
    assert entries[1]['episode'] is None


def test_a_matched_panel_reads_as_a_verdict(report):
    entries = gallery_entries(report['references'], report['episodes'], tolerance=3)
    outcomes = pd.DataFrame(index=[])

    rv = caption(entries[0], outcomes)

    assert "ref Jan 10 2020" in rv
    assert "detected Jan 10 2020 (+0d)" in rv
    assert "day 4 of the attempt" in rv


def test_a_missed_panel_says_what_the_machine_was_doing_instead(report):
    entries = gallery_entries(report['references'], report['episodes'], tolerance=3)

    rv = caption(entries[1], pd.DataFrame(index=[]))

    assert "missed" in rv
    assert "UPTREND" in rv


def test_a_panel_covers_its_episode_with_context_on_each_side(report):
    bars = report['ohlcv'].index
    bar_of = {day: i for i, day in enumerate(bars)}
    entries = gallery_entries(report['references'], report['episodes'], tolerance=3)

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
