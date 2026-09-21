---
name: download-market-data
description: Download or refresh daily OHLCV price history from Yahoo Finance into this project's `md/` market data store using `tools/download_md.py`. Use this whenever the user wants price data for a ticker or index — "download QQQ", "get me Nasdaq Composite history", "update the market data", "refresh SPY through today", "I need ^VIX data since 2010" — or when a backtest, strategy, or signal references a ticker that has no `md/<TICKER>.csv` yet. Use it especially before writing any new download script: the tool already exists, and a naive `yfinance` call from this machine fails on a TLS certificate error that looks like a library bug but is not.
---

# Downloading market data

`tools/download_md.py` is the only supported way to get data into the `md/` store. It
wraps yfinance, normalizes the columns to what `Environment.load_ohlcv` expects, and
appends only rows newer than what's already in the file — so it is safe to re-run and
safe to point at a start date that overlaps existing data.

Do not write a new download script. If the tool is missing a capability (a new interval,
a different provider), extend it rather than adding a parallel one — the store's value
comes from every file being produced the same way.

## 1. Resolve the Yahoo ticker first

Yahoo prefixes index symbols with a caret, which is easy to get wrong. Confirm the symbol
before downloading rather than after:

| Instrument | Index | Tradeable ETF |
|---|---|---|
| Nasdaq Composite | `^IXIC` | — |
| Nasdaq-100 | `^NDX` | `QQQ` |
| S&P 500 | `^GSPC` | `SPY` |
| Dow Jones Industrial Average | `^DJI` | `DIA` |
| Russell 2000 | `^RUT` | `IWM` |
| CBOE Volatility Index | `^VIX` | — |

Which one to pick depends on what the data is for, and the two are not interchangeable:

- **Backtesting a strategy** — use the tradeable ETF. Its prices and volume are what an
  order would actually have filled against.
- **Feeding a signal that reads breadth or market context** — use the index. `FTDSignal`
  takes an `index:` parameter (see `strategies/ftd_belowMA.yaml`) precisely so the
  follow-through day can be detected on the broad market while the trade happens in the
  underlying. Read the volume warning below first, though — it decides which index is
  usable.

If the user names an instrument you can't map confidently, say so and confirm rather than
guessing — downloading the wrong symbol produces a plausible-looking CSV that silently
poisons every backtest that reads it.

### Index volume is often not that index's own volume

This matters more here than in most projects, because a follow-through day is defined as a
price gain on *higher volume than the prior day*. If the volume column belongs to a
different index, `FTDSignal` still runs and still emits confident-looking signals — they
are just derived from the wrong market.

Yahoo does not serve a genuine volume series for every index symbol. Measured over 5,462
trading days from 2005 to 2026, **`^RUT` and `^GSPC` report byte-identical volume on
99.3% of days**. `^RUT` cannot be carrying Russell 2000 volume: it claims ~5.9B shares/day
while `IWM`, an ETF holding the entire Russell 2000, trades ~26M.

| Symbol | Volume usable? |
|---|---|
| `^IXIC` | Yes — genuine Nasdaq composite volume. This is the classic FTD series. |
| `^DJI` | Its own distinct figure. |
| `^GSPC` | Shared with `^RUT`; treat as a broad-market aggregate, not S&P-specific. |
| `^RUT` | **No.** Use `IWM` volume for Russell 2000 breadth instead. |

Before relying on any index's volume in a signal, confirm it isn't shared:

```bash
uv run python -c "
import yfinance as yf
v = yf.download(['^RUT','^GSPC'], start='2024-01-01', progress=False)['Volume']
print((v['^RUT'] == v['^GSPC']).mean())"
```

Anything near 1.0 means the two symbols share a series and at most one of them is real.
Prices are unaffected by this — it is only the volume column that gets borrowed, so an
index remains fine for trend and moving-average work regardless.

## 2. Run the download

yfinance reaches Yahoo through libcurl, and this machine sits behind a TLS-inspecting
proxy (Norton). libcurl rejects the intercepted connection unless it is handed the
inspection root certificate, and the resulting `CertificateVerifyError` /
`unable to get local issuer certificate (20)` traceback reads like a broken install. It
isn't — the bundle just has to be set. This is the same reason `pyproject.toml` sets
`system-certs = true` for uv itself.

PowerShell:

```powershell
$env:CURL_CA_BUNDLE = "C:\ProgramData\Norton\Antivirus\wscert.pem"
uv run python tools/download_md.py "^IXIC" --start 1998-01-01 --end 2026-09-18
```

Bash:

```bash
CURL_CA_BUNDLE="C:\ProgramData\Norton\Antivirus\wscert.pem" \
  uv run python tools/download_md.py "^IXIC" --start 1998-01-01
```

If that certificate path no longer exists, don't disable verification — find the current
one. The same certificate is already wired up for Node, so `$env:NODE_EXTRA_CA_CERTS`
(PowerShell) or `echo $NODE_EXTRA_CA_CERTS` (bash) gives you the path in use.

Notes on the invocation:

- Quote the ticker. `^` is an escape character in `cmd.exe`, and an unquoted `^IXIC`
  silently becomes `IXIC`, which is a different (and wrong) symbol.
- `uv run` is required — yfinance lives in the `dev` dependency group, so a bare `python`
  won't find it.
- `--start` is required; `--end` defaults to today and is inclusive.
- Several tickers can be passed at once: `... download_md.py QQQ SPY --start 2025-01-01`.
- **If the market is still open, set `--end` to the last completed session.** Yahoo serves
  the current day as a partial bar whose high, low, close and volume are still moving.
  Because the tool only appends rows *newer* than the file's last date, a partial bar that
  lands in the store is never corrected by a later run — it stays there as a permanently
  wrong day. A today-row whose volume is a fraction of neighbouring days (a few hundred
  million against a usual several billion) is the tell.
- `--dry-run` prints the rows that would be appended without writing. Reach for it when
  the user is unsure of a date range or symbol, since it makes the mistake visible before
  it lands in the store.

## 3. Verify before reporting success

The tool prints an append summary (`md\^IXIC.csv: appended 7222 rows (1998-01-02 .. 2026-09-18)`),
but check the file itself — an empty or misaligned download still prints cheerfully:

```bash
wc -l "md/^IXIC.csv" && head -3 "md/^IXIC.csv" && tail -3 "md/^IXIC.csv"
```

You're looking for the header `Date,Open,High,Low,Close,Volume`, a first row at or near
the requested start, and a last row at the requested end. Roughly 252 rows per year of
history is the sanity check — substantially fewer means the range was clipped, usually
because the symbol doesn't have history that far back.

On the final row's volume, the direction of the anomaly is what tells you whether to
worry. Abnormally *low* — a fraction of neighbouring days — means a partial bar. Abnormally
*high* is usually legitimate: quarterly triple/quad-witching expirations (third Friday of
March, June, September, December) routinely run 1.5–2x a normal session, so 2026-09-18
printing ~8.9B against a ~5B baseline is the calendar, not a defect.

If a partial bar did get written, fix it by truncating that row from the CSV rather than
re-running the tool; the append-only logic will not overwrite a date it has already seen.

## What the store guarantees

`Environment.load_ohlcv` reads `md/<TICKER>.csv` with the columns
`Date,Open,High,Low,Close,Volume` and nothing else, which is exactly what this tool
writes. Prices are split-adjusted but *not* dividend-adjusted (`auto_adjust=False`),
which is the right choice for signal work: dividend adjustment retroactively rewrites
historical prices every time a dividend is paid, so moving averages and breakout levels
would shift under you between runs.

`load_ohlcv` does have a fallback that auto-downloads when a file is missing, but it
goes through `pandas-datareader`'s long-broken Yahoo endpoint and only reaches back to
2010. Treat a missing file as a cue to run this tool, not as something the backtest will
handle.
