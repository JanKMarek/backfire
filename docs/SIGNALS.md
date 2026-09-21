# Signal definitions

## Follow Through Day (FTD) Signal

### Signal intent

The Follow-Through Day (FTD) signal is a technical chart pattern applied to daily charts of major indices. The signal detects an index turnaround: the market correction (or bear market) is over and a new market uptrend has begun. The signal is binary (True/False). It is True only on the follow-through day itself, and at most once per rally attempt.

All day counts are in trading days. Intraday lows and highs are used to locate the correction low and measure the decline; closing prices are used for everything else (up/down days, the undercut test, the FTD gain).

**Correction precondition.** The signal only looks for a rally attempt while the market is in a correction: the Day 0 low (below) must be at least 8% under the peak, and the peak must be at least 4 weeks (20 days) before Day 0. The peak is the highest intraday high since the most recent FTD that has not failed (see Re-arming), or since the start of the data if there is none.

The signal then has three stages:

- **Day 0 – the rally low.** A day whose intraday low is the lowest intraday low of the last 5 days (that day included) and which satisfies the correction precondition. The Day 0 intraday low is the *rally low*. Day 0 is only searched for while no rally attempt is active; until Day 1 arrives, any later day that makes a lower intraday low becomes the new Day 0, even if it closes up.
- **Day 1 – First Day of the Rally (FDR).** The first day after Day 0 that closes higher than the previous day's close. It begins the rally attempt. From here on the rally low is fixed.
- **Follow-Through Day (Day 4 through Day 25, inclusive).** The first day in this range on which the index closes at least 1.25% above the previous close on volume higher than the previous day's volume. This is the FTD and it confirms the rally attempt. The signal is True on this day, the rally attempt ends, and the market is considered to be in an uptrend. Days 2 and 3 can never be an FTD, however strong.

A rally attempt ends without an FTD, and the search for a new Day 0 resumes, when either:

- **Undercut.** On any day from Day 2 until the FTD, the index *closes* below the rally low. An intraday dip below the rally low with a close at or above it does not end the attempt and does not move the rally low. The undercutting day may itself be the next Day 0 if it qualifies.
- **Timeout.** No FTD has occurred by the end of Day 25.

**Re-arming.** After an FTD the signal stays False until the market is in a correction again. That happens in one of two ways:

- **Failed FTD.** The index closes below the rally low of the confirmed rally. The FTD is marked as failed, the original correction is considered still in force (its peak is used again for the correction precondition), and the next qualifying low is a new Day 0.
- **New correction.** The index declines far enough from the highest high reached since the FTD to satisfy the correction precondition again.

**Parameters.** The classic definition leaves a good deal to discretion. Each parameter below has a default (the permissive end of the customary range) and the range within which it is normally tuned:

| Parameter | Default | Customary range |
|---|---|---|
| Minimum decline from peak to Day 0 low | 8% | 8%–10% |
| Minimum time from peak to Day 0 | 4 weeks (20 days) | 4–6 weeks |
| Day 0 new-low window (intraday lows) | 5 days | 5–10 days |
| Minimum FTD gain (close over previous close) | 1.25% | 1.25%–1.5% |
| Earliest FTD day | Day 4 | fixed |
| Latest FTD day | Day 25 | fixed |

**Reference follow-through days.** The dates below are IBD-style discretionary calls made on the Nasdaq Composite. They are reference points for sanity-checking the signal, not exact expectations: the mechanical rules above may label Day 1 differently (several of the calls count the low day itself as Day 1) or confirm a few days earlier or later, and volume on an ETF proxy such as QQQ does not always agree with index volume.

| Episode | Day 1 of attempted rally | FTD | Notes |
|---|---|---|---|
| Dot-com crash 1, 2001 | Apr 5, 2001 | Apr 10, 2001 | Day 4 off Apr 5. |
| Post 9/11, 2001 | Sep 21, 2001 | Oct 3, 2001 | Off the Sep 21 9/11 low |
| Dot-com crash 2, 2002 | N/A | Oct 15, 2002 | Off the Oct 2002 bear market low. |
| Dot-com crash 3, 2003 | Mar 12, 2003 | Mar 17, 2003 | Day 4 off the Mar 12 low. |
| GFC | Mar 9, 2009 | Mar 12, 2009 | Day 3 off of Mar 9 low. IBD made exception to the Day 4 rule. |
| Sep 2010 | N/A | Sep 1, 2010 | Off the late-Aug 2010 low |
| Xmas 2018, 2018 | Dec 24, 2018 | Jan 4, 2019 | |
| Covid crash | Mar 23, 2020 | Apr 2, 2020 | Covid |
| 2023 AI/tech rally | Dec 28, 2022 | Jan 6, 2023 | |
| Late 2023 | Oct 27, 2023 | Nov 2, 2023 | ??? |
| 2025 | Apr 7, 2025 | Apr 22, 2025 | |
| 2026 | Mar 31, 2026 | Apr 8, 2026 | ??? |

More FTDs where we don't have the exact date but we are pretty sure FTD happened roughly at that time: 
Oct 15, 1998 - surprise Fed cut after oct 8 LCTM low

**Candidate FTD test cases.** Further dates recalled (unverified) as IBD follow-through calls. They are good candidates for FTD test cases, but each date must be confirmed against IBD's archive before it is used as an expectation. Confidence is in the recollection of the exact date: *medium* = fairly sure, *low* = an FTD happened around then but the date may be off, *contradicted* = the recalled date fails a plausibility check on QQQ, so the real FTD is a nearby day. The QQQ check column shows the QQQ close-to-close gain and whether QQQ volume rose on the recalled date.

| FTD (recalled) | Confidence | Context | QQQ check |
|---|---|---|---|
| Jan 17, 1991 | low | Start of Gulf War air campaign | before QQQ data |
| May 25, 2004 | low | Off the May 17 low | +2.0%, vol up |
| Aug 18, 2004 | medium | Off the Aug 13 low | +1.8%, vol up |
| Aug 15, 2006 | medium | Off the Jul 2006 low | +2.5%, vol up |
| Mar 21, 2007 | low | Fed day, after the Feb 27 sell-off | +1.9%, vol up |
| Aug 29, 2007 | medium | Off the Aug 16 low | +2.9%, vol up |
| Mar 20, 2008 | low | Off the Mar 17 low; failed | +2.0%, vol down |
| Mar 1, 2010 | low | Off the Feb 5 low | +1.4%, vol up |
| Jul 7, 2010 | low | Off the Jul 1 low | +3.2%, vol down |
| Aug 23, 2011 | low | Off the Aug 9 low; failed | +4.1%, vol down |
| Oct 12, 2011 | contradicted | Off the Oct 4 low. QQQ only +0.5% on Oct 12; the real FTD is probably Oct 10 | +0.5%, vol up |
| Dec 20, 2011 | low | Off the Nov 25 low | +3.0%, vol up |
| Oct 21, 2014 | medium | Off the Oct 15 low | +2.6%, vol up |
| Oct 5, 2015 | low | After the Aug 24 low and late-Sep retest | +1.4%, vol down |
| Feb 17, 2016 | medium | Off the Feb 11 low | +2.3%, vol up |
| Feb 14, 2018 | low | Off the Feb 9 low | +1.9%, vol up |
| Nov 7, 2018 | medium | Post-midterms, off the Oct 29 low; failed | +3.1%, vol up |
| Jun 5, 2019 | contradicted | Off the Jun 3 low. QQQ only +0.75% on Jun 5 (and only Day 2); the real FTD is a nearby day in early June | +0.75%, vol down |
| Nov 4, 2020 | low | Day after the US election, after the Sep–Oct correction | +4.5%, vol up |
| Oct 14, 2021 | medium | Off the Oct 4 low | +1.8%, vol up |
| Mar 18, 2022 | low | Off the Mar 14 low; failed | +2.1%, vol up |
| May 26, 2022 | low | Off the May 20 low; failed | +2.8%, vol down |
| Jun 24, 2022 | medium | Off the Jun 16 low; failed later | +3.4%, vol up |
| Oct 21, 2022 | medium | Off the Oct 13 low | +2.3%, vol up |
| Jan 6, 2023 | medium | Off the Dec 28 low (also in the reference table above) | +2.8%, vol up |
| Nov 1 / Nov 2, 2023 | low | Off the Oct 26 low; Nasdaq and S&P 500 possibly called on different days (see reference table above) | Nov 1: +1.7%, vol up; Nov 2: +1.8%, vol down |
| Aug 13, 2024 | low | Off the Aug 5 low | +2.5%, vol up |
| Mar 24, 2025 | low | Off the Mar 13 low; failed | +2.0%, vol down |
| Apr 22, 2025 | medium | Off the Apr 7 low (also in the reference table above) | +2.6%, vol up |

