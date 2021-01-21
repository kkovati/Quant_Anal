import numpy as np
import pandas as pd


def rand_price_sampler(df, pre, post):
    # pre interval includes current day
    # e.g pre = 2 : yesterday and today is used for prediction input

    # date = random.choice(df.index[pre-1:len(df)-post])
    
    # random start index
    start = np.random.randint(len(df) - pre - post + 1)
    # list of analyzed dates of days
    days = df.index[start : start + pre + post]
    # random symbol from columns
    symbol = np.random.choice(df.columns)
    
    # Series of prices
    prices = df.loc[days, symbol]

    if not all(prices >= 0):
        print(prices)
        raise Exception('Problem with the prices')
        
    return prices
        
    
if __name__ == '__main__':
    
    df = pd.read_csv('iex_cloud_connector\\test.csv')
    # 'D:\\Kovacs_Attila\\08_Programming\\Python_projects\\Quant_Anal\\iex_cloud_connector\\test.csv')
    
    df.set_index('Date', inplace=True)
    # print(df)
    
    prices = rand_price_sampler(df, 3, 2)
    
    print(prices)
                                
    