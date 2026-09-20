# Signal definitions

## Follow Through Day (FTD) Signal

### Signal intent. 

The Follow-Through Day (FTD) signal is a technical chart pattern applied to major indices. It confirmst that a new market uptrend has begun after a correction or a bear market. Conceptually, the signal has three stages: 
- Rally Attempt (Day 1). After a decline or correction when a market has hit a new low, the market closes higher than a prior day. This begins the really attempt. 
- Days 2 and 3. The index close must not undercut the Day 1 low. 
- The Follow-Through (Day 4 and beyond, out to day 25). To confirm the uptrend, the index must gain at least 1.25%-1.5% on the day on total volume higher than previous day volume. At this point, the market is considered in uptrend. 

The uptrend ends when the index closes below the low of Day 1 or when it retraces by 10% from previous highs. 

The FTD signal is considered failed if it breaches the low of Day 1 before it retraces by 10% from previous highs. Otherwise, it is considered successful. 

There is a significant amount of discretion in formalizing this intent. The usual values are: initial decline at least 8%-10% (and should take 4-6 weeks), the new low should be rached by intraday price and is measured over a 5-10 day window. 

Standard follow-through days (all on Nasdaq): 

Dot-com Crash Failed FTD: Day 1 of attempted rally Apr 4, 2001, FTD Apr 18, 2001. Failed FTD. 
Dot-com Crash FTD: Day 1 of attempted rally Mar 13, 2003, FTD Mar 21, 2003. Successful.  
GFC Crisis FTD: Day 1 of attempted rally Mar 10, 2009, FTD Mar 18, 2009. Successful. 
Covid Crash FTD: Day 1 of attempted rally mar 23, 2020, FTD date Apr 2, 2020. Successful FTD. 
2023 AI/Tech Rally: Day 1 of attempted rally Dec 28, 2022, FTD Jan 6, 2023. Succesful FTD. 
Late 2023 FTD: Day 1 of attempted rally Oct 27, 2023, FTD Nov 2, 2023. Sucessful. 
2025 FTD: Attmpted rally day 1: Apr 9, 2025. FTD Apr 23, 2025. Successful. 
2026 FTD: Attmpted rally day 1: Mar 31, 2026. FTD Apr 8, 2026. Successful. 

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

The signal is True when it is ON and False when it is OFF or RALLY_ATTEMPT. 


