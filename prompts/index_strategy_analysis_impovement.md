# Index Strategy Analysis and Improvement

## Start with a popular and obvoius choice of index strategy - MA crossover. 
Simulate strategy ma_crossover with short MA = 50d and long MA = 200d. Risk management set to 7% stop loss. Simulate on QQQ on period 2000 - 2025. 

Simulation shows total return 818% vs. buy and hold strategy return approx. 500% over the same period. There were relativly few trades so the strategy is appropriate for end of the day trader. The three largest max drawdowns around -29% and correspond to the three major crashes in that period: dot-com bust, the Great Financial Crisis and the Covid crash. 

## Analyze the trades of this strategy. 
We make the following observations: 
- the strategy should make money in uptrends and avoid large market corrections. In general, it does do that. 
- some actions act as "paying the price for avoiding potential crash" or "paying insurance premiums": those are the trades where the sell order is below the following buy order, so in effect we sell and then buy back the position at a higher price. So we give away some profits for avoiding the potential crash. There are 13 such "paying insurance" trades in the simulation window. 
- other actions act as "taking advantage of market crashes": those are the trades where the sell is above the next buy, so effectively, we buy back the position at a lower price, taking advantage of the dip. There are 3 such trades corresponding to the three major market crashes in the simuation window
- the strategy probably outperforms the buy-hold strategy because of the large dot-com bust which allowed us to take advantage of the large dip

## Identify possible improvement approaches
One approach is to exit the position earlier and reenter earlier. This could reduce the cost of eliminating the drawdnowns even thoug hit may lead to more active trading and more false breakouts. We need to use more sensitive signals:  
- Exit on close below 21d/50d MA and reenter of Follwo-Through Day. Note that this is essentially Vibha Jha's strategy. 
- Use MA crossover but shorter MA periods. Explore several short/long MA period combinations.  
Another approach is to attempt to predict whether the pullback is a minor/regular pullback or a major crash. 
- How would we do that? 