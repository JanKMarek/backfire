# Signal definitions

## Follow Through Day (FTD) Signal

### Signal intent

The Follow-Through Day (FTD) signal is a technical chart pattern applied to daily charts of major indices. The signal is intended to detect index turnarounds after a market correction; it is designed to detect the turnaround early and without undue lag; the inevitable false positives will be handled by risk management measures (stop loss). As such, it can be used to identify beginning stages of trends which start after a correction. It will not work very well for trends which start out of a rangebound market period (no preceding correction so a channel breakout siganl might work better there) or which start with a strong volatility explosion (because it waits several days before the trend is confirmed). 

The signal starts looking for a rally attempt after a correction which is at least 20 days long and 8% deep. Once such a correction has been detected, it marks the low of the correction, then looks for the start of the rally (an up day) and then attempts to confirm this rally with a strong up-day which follows several days later. 

### Signal specification

The signal is binary (True/False), True only on the turnaround day itself, and at most once per rally attempt.

All day counts are in trading days. Intraday lows and highs are used to locate the correction low and measure the decline; closing prices are used for everything else (up/down days, the undercut test, the FTD gain).

The signal only looks for a rally attempt afther the market has corrected/pulled back: the Day 0 low (below) must be at least 8% under the peak, and the peak must be at least 4 weeks (20 days) before Day 0. The peak is the highest intraday high since the most recent FTD that has not failed (see Re-arming), or since the start of the data if there is none.

The signal then has three stages:

- **Day 0 – the rally low.** A day whose intraday low is the lowest intraday low of the last 5 days (that day included) and which satisfies the correction precondition. The Day 0 intraday low is the *rally low*. Day 0 is only searched for while no rally attempt is active; until Day 1 arrives, any later day that makes a lower intraday low becomes the new Day 0, even if it closes up.
- **Day 1 – First Day of the Rally (FDR).** The first day after Day 0 that closes higher than the previous day's close. It begins the rally attempt. From here on the rally low is fixed. Note that the FDR does not need to follow the rally low day immediately. 
- **Follow-Through Day (Day 4 through Day 25, inclusive).** The first day in this range on which the index closes at least 1.25% above the previous close on volume higher than the previous day's volume. This is the FTD and it confirms the rally attempt. The signal is True on this day, the rally attempt ends, and the market is considered to be in an uptrend. Days 2 and 3 can never be an FTD, however strong.

A rally attempt ends without an FTD, and the search for a new Day 0 resumes, when either:

- **Undercut.** On any day from Day 2 until the FTD, the index *closes* below the rally low. An intraday dip below the rally low with a close at or above it does not end the attempt and does not move the rally low. The undercutting day may itself be the next Day 0 if it qualifies.
- **Timeout.** No FTD has occurred by the end of Day 25.

**Re-arming.** After an FTD the signal stays False until the market is in a correction again. That happens in one of two ways:

- **Failed FTD.** The index closes below the rally low of the confirmed rally. The FTD is marked as failed, the original correction is considered still in force (its peak is used again for the correction precondition), and the next qualifying low is a new Day 0.
- **Successful FTD Ending in a New Correction.** The index advances and then, at some point, declines far enough from the highest high reached since the FTD to satisfy the correction precondition again. This means the advance must be more than 8% off of the low of the confirmed rally.  

### Signal parameters

**Parameters.** The classic definition leaves a good deal to discretion. Each parameter below has a default (the permissive end of the customary range) and the range within which it is normally tuned:

| Parameter | Default | Customary range |
|---|---|---|
| Minimum decline from peak to Day 0 low | 8% | 8%–10% |
| Minimum time from peak to Day 0 | 4 weeks (20 days) | 4–6 weeks |
| Day 0 new-low lookback window (intraday lows) | 5 days | 5–10 days |
| Minimum FTD gain (close over previous close) | 1.25% | 1.25%–1.5% |
| Earliest FTD day | Day 4 | fixed |
| Latest FTD day | Day 25 | fixed |

**What the signal records.** Besides `es`, the signal writes a row of diagnostics for every day so that a run can be audited without re-deriving it. Next to the state and the peak, Day 0, rally low and rally day it is working with, three columns say what the machine did with the day and why:

| Column | Values | Meaning |
|---|---|---|
| `event` | `DAY0`, `DAY0_LOWERED`, `DAY1`, `UNDERCUT`, `TIMEOUT`, `FTD`, `FTD_FAILED`, `NEW_CORRECTION` | The transitions the day made, joined with `+` when it makes more than one, e.g. `UNDERCUT+DAY0`. `es` is on exactly on the days whose event contains `FTD`. |
| `day0_block` | `DECLINE`, `PEAK_AGE`, or both | On a day that makes a new Day 0 window low but is not a Day 0: the clauses of the correction precondition it fails. |
| `ftd_block` | `TOO_EARLY`, `VOLUME`, or both | On a rally day that gains at least the minimum but is not a follow-through day: the remaining clauses it fails. |

None of this feeds back into the rules. Like `es`, these three mark a single day: they are never carried over to the next one, and a mark made on a day the underlying did not trade moves to its next trading day.

### Signal verification and analysis

Signal verification and analysis is performed by running ftd_signal_verification_report (see CLI Workflows in README.md). 

We will verify the signal specification and implementatoin against two types of ground truth:  
1. **Does the signal meet the intent**? File docs/turnaround_points.csv contains a set of turnaround points that occurred in the market (IXIC) since 1999 and we would expect to be detected by the FTD signal. Note that these are not official IBD FTD calls, these turnaround points were selected visually with the benefit of hindsight (i.e., looking at chart with some data right of the FTD point). 
1. **Does the signal fire when IBD calls a FTD**? File docs/ftd_reference.yaml contains a list of IBD calls we could obtain.  

A positive is a date when the FTD signal fires.  
True positive is a firing date that matches a ground truth date
False positive is a firing date that does not match a ground truth date

Note that matching of ground truth dates and positives (dates when FTDSignal fires) is done with some tolerance since the signal may fire a few days earlier or later. The tolerance is at most 1 business day earlier and up to 3 business days later. These tolerances are verification report parameters. 

The verification report lists: 
- signal backtest run parameters and verification test paramters
- statistics: 
  - ground truth dates count, positives count, number of trading days in the simulation range
  - precision: true_positives / (true_positives + false_positives)
  - recall: true_positives / (ground truth dates) 
  - F1 metric
- scorecard: table listing the following for each ground truth date: 
  - ground truth date
  - signal firing date (blank if not matched)
  - reason for not firing (if not matched) 
  - note for the ground truth date
  - the false positives are listed in the same table, and all rows are in chronological order
- average underlying paths after firing (for up to 60 days after) for all FTDs, successful FTDs and failed FTDs
- statistics for underlying path after successful and failed FTDs: median @ 5days, 10days, 20days, 40days, 60days
- gallery: contains explanatory charts for all episodes (ground truth episodes as well as false positives). The charts are linkable from the scorecard table above
  - explanatory chart for the episode

### Signal analysis results - summary

Simulating the FTD signal over the period 1999-2026 shows using standard parameters shows: 
- we hand-marked 71 turnaround points on the IXIC daily chart over this period
- signal fired 74 times (~ 3 times a year). 
- turnaround points: 48 were matched against signal firings, 23 were missed. 10 misses were due to no preceding correction (so the market was still in uptrend), 8 fired too late (due to no high volume strong up day).  
- signal firings: 48 were matched against ground truth dates, 26 were false positives. Some false positives were due to FTD price being close to the FDR low (and so visually these were not marked as turnaround points).  
- qualitatively, signal captures well turnarounds after a correction. It does not do so well to capture two other types of trend starts: breakouts from sideways markets (due to lack of correction) and volatility explosions (since there may not be a strong up day on high volume to confirm the direction). 
- of the 73 FTD firings, 23 failed later and 50 ended in a new correction
- most missed FTDs missed due to min_peak_age_days=15 (often the peak was picked since the last FTD and so was too close. use window?)

Notes on recent firings: 
  - matched 11 true positives - all major (hand-picked) trend starts since the Covid crash 
  - produced 3 false positives: FTD occurred on days 21, 16, 19 (so late), 
  - produced 1 miss: Nov 10, 2022, still in uptrend (only looking back to prev Oct 21, 2022 FTD so not enough correction)
  - produced 1 miss: Mar 11, 2022, no corr'n due to Day 0 happening 13 trading days after peak
In more distant past, there were false positives due to the FTD happening 

Note: no confidence intervals, no train/validate/test split of training data, 
  - forward return table shows practically no edge over the market
  - ground truth is signal aware (handpicking not turnaround points but FTD points)
  - 




