# Signal definitions

## Follow Through Day (FTD) Signal

### Signal intent. 

The Follow-Through Day (FTD) signal is a technical chart pattern applied to daily charts of major indices. The signal detects index turnaround - the market correction (or bear market) is over and a new market uptrend has begun. The signal is a binary signal (True/False) and fires on the follow-through day only. 

Conceptually, the signal has three stages: 
- First Day of the Rally (Day 1). After a decline or correction, the market has hit a new low (Day 0) and the next day, the market closes higher than a prior day. This up day is the FDR and begins the rally attempt. The rally low is the lowest low of the correction up to and including Day 1 (usually the Day 0 low). 
- Days 2 and 3. The index close must not undercut the rally low. If it does, the rally attempt is over.  
- The Follow-Through Day (Day 4 and beyond, out to day 25). The index gains at least 1.25%-1.5% on this day on total volume higher than previous day volume. This is the FTD and it 'confirms' that rally attempt. The siganl indicates True on this day. At this point, the market is considered in uptrend. 
- If no FTD occurs within 25 days of the FDR, the rally attempt is considered over. 

As you can see, there is a significant amount of discretion in the above signal definition. The standard parameter values are: initial decline marking the correction/bear market is at least 8%-10% (and should take 4-6 weeks). The new low on Day 0 should be reached by intraday price or closing price and is measured over a 5-10 day window of closing prices. The FTD should occur between days 4 and 25 (inclusive) and the index gain should be 1.25%-1.5%.   

Standard follow-through days (all on Nasdaq): 

Dot-com Crash 2001 FTD: Day 1 of attempted rally Apr 4, 2001, FTD Apr 18, 2001. 
Dot-com Crash 2003 FTD: Day 1 of attempted rally Mar 13, 2003, FTD Mar 21, 2003. 
GFC Crisis FTD: Day 1 of attempted rally Mar 10, 2009, FTD Mar 18, 2009. 
Covid Crash FTD: Day 1 of attempted rally mar 23, 2020, FTD date Apr 2, 2020. 
2023 AI/Tech Rally: Day 1 of attempted rally Dec 28, 2022, FTD Jan 6, 2023. 
Late 2023 FTD: Day 1 of attempted rally Oct 27, 2023, FTD Nov 2, 2023. 
2025 FTD: Attmpted rally day 1: Apr 9, 2025. FTD Apr 23, 2025. 
2026 FTD: Attmpted rally day 1: Mar 31, 2026. FTD Apr 8, 2026. 

### Signal formalization: 

Parameters and their default values: 
- index: default IXIC
- lookback_days: 10 (business days) 
- ftd_min_days: 4 (business days)
- ftd_min_gain: 0.015
- ftd_max_days: 25 (business days)
- terminal_retracement: 0.1

Logic: Starts in OFF state, and each day is processed as follows: 

When OFF: 
  - the correction low is the lowest low since the signal went OFF (the day it went OFF included)
  - if index closes higher than the prior day after making a new lookback_days day low since the signal went OFF, set state to RALLY_ATTEMPT; this is day 1 of the rally attempt and the rally low is the correction low as of day 1
  - otherwise stay at OFF
When RALLY_ATTEMPT: each day is day N of the rally attempt, where N counts business days from day 1 (the day after day 1 is day 2, so with ftd_min_days = 4 the earliest Follow Through Day is the fourth day of the attempt, as in IBD). The rules are checked in order and the first that matches wins:
  - if index closes below the rally low, set state to OFF (the rally attempt failed)
  - if N >= ftd_min_days and index has daily gain at least ftd_min_gain with volume greater than previous day, set state to ON (this day is the Follow Through Day)
  - if N >= ftd_max_days, set state to OFF (the rally attempt expired without a Follow Through Day)
  - otherwise stay at RALLY_ATTEMPT
A failed or expired rally attempt does not need another lookback_days day low: the next day that closes higher than the prior day is day 1 of a new rally attempt, with the rally low being the lowest low since the signal went OFF. Only a Follow Through Day (state ON) ends the sequence of rally attempts, and after the signal goes OFF again, a new lookback_days day low is needed before the first rally attempt.
When ON:  
  - on index close retracement more than terminal_retracement from the highest close since the signal turned ON, set state to OFF

The signal is True only on the Follow Through Day, i.e. the day the state changes from RALLY_ATTEMPT to ON. It is False on every other day: in OFF, in RALLY_ATTEMPT, and on the remaining days of ON. The ON state means the market is considered in uptrend; it fires no further signal and only blocks new rally attempts until the terminal retracement sets the state back to OFF.

The signal value (True/False) is distinct from the state (OFF/RALLY_ATTEMPT/ON): "the signal went OFF" and "the signal turned ON" above refer to the state changing from ON to OFF and from RALLY_ATTEMPT to ON, not to the signal value changing. 


