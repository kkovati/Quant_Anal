import pandas as pd 
import requests


def query_history():
    """
    Queries all available close price of a single symbol
    Symbols read fro

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    stocks = pd.read_csv('data_mining\sp_10_stocks.csv')
    symbols_list = stocks['Symbol'].tolist()  
    
    df = pd.DataFrame(columns = symbols_list)
    df.index.name = 'Date'
    
    for i, symbol in enumerate(symbols_list):
        print(f'Request {symbol} {i+1}/{len(symbols_list)}')
        
        url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/chart/max?chartCloseOnly=true&token={token}'
        
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f'HTTP Response Status Code: {response.status_code}')
        data_json = requests.get(url).json()
    
        for daily_data in data_json:
            date = int(daily_data['date'].replace('-',''))
            df.at[date, symbol] = daily_data['close'] 
    
    df.sort_index(inplace=True)
    
    #print(df)    
    
    df.to_csv(path_or_buf='data_mining\temp.csv')


if __name__ == '__main__':
    
    query_history()