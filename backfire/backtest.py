"""
    Command line driver for a single backtest run.

    Reads a strategy definition from a YAML file, builds the corresponding
    SignalDrivenStrategy and runs Strategy.backtest over one underlying and date range:

      uv run python backfire/backtest.py --start_date "2020-01-01" --end_date "2026-07-01" \
          -md "md" --underlying QQQ --strategy strategies/ma_crossover.yaml --out "out/my_strategy"

    The strategy file names the component classes and their constructor arguments. Every
    Signal / BasicRiskManagement / PositionManagement subclass defined in backfire.base
    and backfire.signals can be used by its class name:

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

    An argument that is itself a mapping with a 'name' key is built as a nested signal,
    which is how signal combinators are configured:

      exit_signal:
        name: ReverseSignal
        signal:
          name: ShortMAAboveLongMA
          short_MA: 50
          long_MA: 200

    A combinator taking any number of signals is given a list of them:

      exit_signal:
        name: OrSignal
        signals:
          - name: ShortMABelowLongMA
            short_MA: 50
            long_MA: 200
          - name: BreakBelowMA
            period: 20

    Individual entries can be overridden from the command line without editing the file:

      --set strategy.entry_signal.short_MA=20 --set strategy.risk_management.stop_loss=null
"""

import argparse
import inspect
import math
import os
import sys
from datetime import datetime

import yaml

from backfire import base, signals
from backfire.base import (
    BasicRiskManagement,
    Environment,
    PositionManagement,
    Signal,
    SignalDrivenStrategy,
)

_COMPONENT_MODULES = (base, signals)

# Dollar amounts are reported with no decimals and a thousands separator, everything
# else that is fractional (returns, percentages, holding periods) with two decimals.
_MONEY_STATS = ('total_pnl', 'positive_pnl', 'negative_pnl')


def _catalog(*component_bases):
    """
        Collects the strategy components that can be named in a strategy file.
    :param component_bases: base classes to look for (e.g. Signal)
    :return: dict of class name -> class, over the classes defined in backfire.base
             and backfire.signals
    """
    rv = {}
    for module in _COMPONENT_MODULES:
        for name, obj in vars(module).items():
            if (inspect.isclass(obj)
                    and obj.__module__ == module.__name__
                    and issubclass(obj, component_bases)):
                rv[name] = obj
    return rv


SIGNALS = _catalog(Signal)
RISK_MANAGEMENTS = _catalog(BasicRiskManagement)
POSITION_MANAGEMENTS = _catalog(PositionManagement)


def build_component(spec, catalog, kind):
    """
        Instantiates one strategy component from its strategy file section.
    :param spec: mapping with a 'name' key naming the class, the remaining keys are
                 passed to the constructor. None builds nothing.
    :param catalog: dict of class name -> class the 'name' is looked up in
    :param kind: what is being built, used in error messages
    :return: the component instance, or None if spec is None
    """
    if spec is None:
        return None
    if not isinstance(spec, dict) or 'name' not in spec:
        raise ValueError(f"The {kind} must be a mapping with a 'name' key, got: {spec!r}")

    kwargs = dict(spec)
    class_name = kwargs.pop('name')
    if class_name not in catalog:
        raise ValueError(f"Unknown {kind} '{class_name}'. "
                         f"Available: {', '.join(sorted(catalog))}")

    # an argument that is itself a named mapping is a nested signal (e.g. ReverseSignal),
    # a list of them a list of nested signals (e.g. OrSignal)
    kwargs = {k: _build_nested_signals(v) for k, v in kwargs.items()}

    # a list given under the name of the constructor's *args parameter supplies the
    # positional arguments, so that 'signals: [...]' configures OrSignal(*signals)
    cls = catalog[class_name]
    args = ()
    for param in inspect.signature(cls).parameters.values():
        if param.kind is param.VAR_POSITIONAL and isinstance(kwargs.get(param.name), list):
            args = tuple(kwargs.pop(param.name))

    try:
        return cls(*args, **kwargs)
    except TypeError as e:
        raise ValueError(f"Cannot build {kind} '{class_name}': {e}") from e


def _build_nested_signals(value):
    """
        Builds the nested signals in one constructor argument of a strategy file: a mapping
        with a 'name' key becomes a signal, a list has each of its entries built the same way,
        anything else is passed through unchanged.
    """
    if isinstance(value, dict) and 'name' in value:
        return build_component(value, SIGNALS, 'signal')
    if isinstance(value, list):
        return [_build_nested_signals(v) for v in value]
    return value


def build_strategy(conf, env):
    """
        Builds a SignalDrivenStrategy from a parsed strategy file.
    :param conf: the parsed strategy file - a mapping with a 'strategy' key
    :param env: the Environment the strategy runs in
    :return: SignalDrivenStrategy
    """
    if not isinstance(conf, dict) or 'strategy' not in conf:
        raise ValueError("The strategy file must contain a top level 'strategy' mapping.")
    s = conf['strategy']
    if not isinstance(s, dict):
        raise ValueError(f"The 'strategy' entry must be a mapping, got: {s!r}")
    if 'entry_signal' not in s:
        raise ValueError("The strategy must define an 'entry_signal'.")

    return SignalDrivenStrategy(
        env=env,
        entry_signal=build_component(s['entry_signal'], SIGNALS, 'entry signal'),
        exit_signal=build_component(s.get('exit_signal'), SIGNALS, 'exit signal'),
        risk_management=build_component(s.get('risk_management'), RISK_MANAGEMENTS,
                                        'risk management'),
        position_management=build_component(s.get('position_management'), POSITION_MANAGEMENTS,
                                            'position management'),
        name=s.get('name'))


def apply_overrides(conf, overrides):
    """
        Applies command line overrides to a parsed strategy file, in place.
    :param conf: the parsed strategy file
    :param overrides: iterable of "dotted.path=value" strings, the path being relative to
                      the root of the file, e.g. "strategy.entry_signal.short_MA=20".
                      The value is read as YAML, so 20, 0.07, null and true keep their types.
    :return: conf
    """
    for override in overrides:
        path, sep, raw_value = override.partition('=')
        if not sep:
            raise ValueError(f"Override '{override}' is not of the form path=value.")
        keys = [k for k in path.strip().split('.') if k]
        if not keys:
            raise ValueError(f"Override '{override}' has an empty path.")

        node = conf
        for i, key in enumerate(keys[:-1]):
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot apply override '{override}': "
                                 f"'{'.'.join(keys[:i + 1])}' is not a mapping.")
        node[keys[-1]] = yaml.safe_load(raw_value)
    return conf


def load_strategy_conf(path, overrides=()):
    """
        Reads a strategy file and applies command line overrides.
    :return: the parsed strategy file
    """
    with open(path) as f:
        conf = yaml.safe_load(f)
    if not isinstance(conf, dict):
        raise ValueError(f"Strategy file '{path}' does not contain a YAML mapping.")
    return apply_overrides(conf, overrides)


def run_backtest(strategy_conf, underlying, start_date, end_date=None, md="./md", out=""):
    """
        Runs one backtest: builds the strategy described by strategy_conf and backtests it
        over the underlying and the trading period. Evaluation data is written to 'out'
        and returned to the caller.
    :param strategy_conf: parsed strategy file (see load_strategy_conf)
    :param underlying: ticker to trade
    :param start_date: first day of the trading period, YYYY-MM-DD
    :param end_date: last day of the trading period, YYYY-MM-DD, None for the most recent data
    :param md: market data directory
    :param out: output directory, "" for no persistent output
    :return: (strategy, price_and_signals, positions, trades, stats)
    """
    env = Environment(md=md, out_dir=out)
    if env.out_dir:
        # saved with overrides already applied, so it reflects what was actually run
        with open(os.path.join(env.out_dir, "strategy.yaml"), "w") as f:
            yaml.safe_dump(strategy_conf, f, sort_keys=False)
    strategy = build_strategy(strategy_conf, env)
    price_and_signals, positions, trades, stats = strategy.backtest(
        ticker=underlying, from_date=start_date, to_date=end_date)
    return strategy, price_and_signals, positions, trades, stats


def format_stats(stats):
    """
        Renders the performance metrics as an aligned text block.
    """
    width = max(len(str(k)) for k in stats.index)
    lines = []
    for key, value in stats.items():
        if isinstance(value, float) and math.isnan(value):
            shown = "NA"
        elif key in _MONEY_STATS:
            shown = f"{value:,.0f}"
        elif isinstance(value, float):
            shown = f"{value:.2f}"
        else:
            shown = str(value)
        lines.append(f"  {str(key):<{width}}  {shown}")
    return "\n".join(lines)


def format_catalog():
    """
        Renders the catalog of strategy components that can be named in a strategy file
        (see SIGNALS, RISK_MANAGEMENTS, POSITION_MANAGEMENTS).
    """
    sections = [("Signals", SIGNALS), ("Risk managements", RISK_MANAGEMENTS),
               ("Position managements", POSITION_MANAGEMENTS)]
    lines = []
    for title, catalog in sections:
        lines.append(f"{title}:")
        lines += [f"  {name}" for name in sorted(catalog)]
    return "\n".join(lines)


def _valid_date(value):
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a date in YYYY-MM-DD format.")
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="backtest.py",
        description="Backtest a signal driven strategy over one underlying.")
    p.add_argument("--start_date", type=_valid_date,
                   help="first day of the trading period, YYYY-MM-DD")
    p.add_argument("--end_date", default=None, type=_valid_date,
                   help="last day of the trading period, YYYY-MM-DD "
                        "(default: the most recent market data available)")
    p.add_argument("-md", "--md", dest="md", default="./md",
                   help="market data directory (default: ./md)")
    p.add_argument("--underlying",
                   help="ticker the strategy trades, e.g. QQQ")
    p.add_argument("--strategy",
                   help="path to the strategy definition YAML file")
    p.add_argument("--out", default="",
                   help='output folder for the run; "" (the default) keeps no persistent output')
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE",
                   help="override one strategy file entry, e.g. "
                        "--set strategy.entry_signal.short_MA=20. Can be repeated.")
    p.add_argument("--list_signals", action="store_true",
                   help="list all strategy components (signals, risk managements and "
                        "position managements) that can be named in a strategy file, then exit")

    args = p.parse_args(argv)
    if not args.list_signals:
        required = [("--start_date", args.start_date), ("--underlying", args.underlying),
                    ("--strategy", args.strategy)]
        missing = [name for name, value in required if value is None]
        if missing:
            p.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.list_signals:
        print(format_catalog())
        return 0

    try:
        conf = load_strategy_conf(args.strategy, args.overrides)
        strategy, _, _, trades, stats = run_backtest(
            strategy_conf=conf,
            underlying=args.underlying,
            start_date=args.start_date,
            end_date=args.end_date,
            md=args.md,
            out=args.out)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as e:
        print(f"backtest failed: {e}", file=sys.stderr)
        return 2

    print(f"Strategy   : {strategy.name}")
    print(f"Underlying : {args.underlying}")
    print(f"Period     : {args.start_date} .. {args.end_date or 'latest'}")
    print(f"Output     : {args.out or '(none)'}")
    print(f"Trades     : {len(trades)}")
    print("Performance:")
    print(format_stats(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
