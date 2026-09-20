# MARKET DATA APIs

## Requirements
1. Ticker catalog. Allows me to get a list of tickers - e.g., in an index, within sectors.
1. Historical OHLCV data for a ticker going back to at least 1998 (so includes dot com crash)
1. Intraday historical data for a ticker - 1min or 5min resolution suffices. 
1. Get market cap, quarterly revenue growth YoY, quarterly earnings growth YoY. 
1. Earnings and revenue announcement values - to calculate revenue and earnings growth.
1. Company guidance and outlook - raised,lowered?    
1. Analysts' revisions (upgrades/downgrandes) in the days after earnings
1. Company short interest


## APIs: 
I researched this back in 2021, so the landscape may be different today:  
1. FMP 
2. Polygon
3. AlphaVantage
4. Yahoo Finance
5. Google Finance

## FMP
I have not researched it. 

## Polygon 
I used it in the past, meets most (all?) criteria. 

## AlphaVantage
I have not researched it. 

## Yahoo Finance - meets some criteria
Yfinance used to be the package to use in python.

## Google Finance - not suitable
Google Finance stopped providing direct access API in 2011 and the APi is now only available via GoogleSheets. 
There are libraries for accessing the data but they work by web scraping the Google Finance web site. 




