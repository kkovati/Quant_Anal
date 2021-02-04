


# https://en.wikipedia.org/wiki/Technical_analysis

# https://en.wikipedia.org/wiki/Relative_strength_index
# https://en.wikipedia.org/wiki/Trix_(technical_analysis)

# https://en.wikipedia.org/wiki/MACD

# https://towardsdatascience.com/trading-toolbox-02-wma-ema-62c22205e2a9
# https://www.investopedia.com/terms/e/ema.asp
# https://en.wikipedia.org/wiki/Moving_average

def exp_mov_avg(l):
  pass
  
def ema_test(l, days):

  smoothing = 2
  q = smoothing / (1 + days)
  
  output = []
  
  ema_yesterday = 0
  
  for i, value_today in enumerate(l):
    ema_today = value_today * q + ema_yesterday * (1 - q)
  
    if i >= days: # when exactly?
      output.append(ema_today)
     else:
      output.append(nan)
    
    ema_yesterday = ema_today
    
    return output
    
    
if __name__ == "__main__":
  pass
