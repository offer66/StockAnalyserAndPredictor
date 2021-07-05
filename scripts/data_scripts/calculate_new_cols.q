// Load the csv files from datasets/scraped DIR to memory

sym:`AAPL`GME`ABNB`PLTR`ETSY`ENPH`GOOG`AMZN`IBM`DIA`IVV`NIO;
scrapedData:sym!{("zffffi";enlist",") 0: hsym `$"datasets/scraped/", string[x], "/", string[x],"-total-data.csv"}each sym;



// perform col manipulation to get extra data:
// - ema_3          EMA_today = (VALUE_today * (SMOOTHING/ 1 + DAYS) ) + EMA_yesterday * (1 - (SMOOTHING / 1 + DAYS))
// - ema_5          SMOOTHING = 2
// - ema_30
// - sma_30
// - sma_50
// - macd
// - rsi
emaDays:3 5;
update ema3: (2%1+3)ema scrapedData[`AAPL][`open] from scrapedData[`AAPL]
{![`scrapedData[`AAPL]; (); 0b; (enlist `$("ema",string[x]))!enlist((2%1+x)ema scrapedData[`AAPL][`open])]} 3;




