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

## Daily market data 
Daily market data (OHLCV) is cached in the md folder in files named {ticker}.csv. To download data not present in the md folder, use skill download-market-data or script tools/download_md.py. 





