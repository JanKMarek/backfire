# MARKET DATA APIs

## My requirements
1. Get a list of symbols (maybe within certain sectors). This can be obtained by running a screen in TradeView and 
saving tickers in a .csv file.
1. Get market cap, quarterly revenue growth YoY, quarterly earnings growth YoY. 
1. Get OHLCV data for the day of interest. 
1. Get OHLCV data for 21d prior to the day of interest - for calculating ranges, MAs, average volume, etc.
1. Get OHLCV data for 1d, 2d, 3d, 5d, 1month forward - for calculating returns.    
1. Earnings and revenue announcement values - to calculate revenue and earnings growth.
1. Company guidance and outlook - raised,lowered?    
1. Analysts' revisions (upgrades/downgrandes) in the days after earnings
1. Company short interest
1. 


## Current APIs: 
Current APIs ranked by best fit: 
1. FMP 
2. Polygon
3. AlphaVantage
4. Yahoo Finance
5. Google Finance

## FMP 


## Polygon 



## AlphaVantage
Now seems to be worse than Polygon or FMT. 

## Yahoo Finance
Yfinance used to be the package to use in python. As of 2017, yahoo finance started throttling the requests and it 
is a nuisance to use now. 

## Google Finance
Google Finance stopped providing direct access API in 2011 and the APi is now only available via GoogleSheets. 
There are libraries for accessing the data but they work by web scraping the Google Finance web site. 




