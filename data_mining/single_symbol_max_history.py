import pandas as pd 
import requests
from .data_mining.get_request import get_request


def single_symbol_max_history(symbol):
    """
    Queries all available close price of a single symbol
    """
    with open('token\iex_cloud_api_token.txt', 'r') as file:
        token = file.read()       
    
    print(f'Request max history of {symbol}')
    
    url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/chart/max?chartCloseOnly=true&token={token}'
    
    data_json = get_request(url)
    
    df = pd.DataFrame(columns = [symbol])
    df.index.name = 'Date' 

    for daily_data in data_json:
        date = int(daily_data['date'].replace('-',''))
        df.at[date, symbol] = daily_data['close'] 
    
    df.sort_index(inplace=True)
    
    return df


if __name__ == '__main__':
    
    df = single_symbol_max_history(symbol='AAPL')    
    
    assert df.columns[0] == 'AAPL'
    assert len(df.columns) == 1
    assert df.index.name == "Date"
    assert df.index[0] > 20060000
    assert len(df.index) > 3500
    assert len(df) > 3500
    
    print("Test1 OK")
    