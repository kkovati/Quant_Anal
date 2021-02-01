import numpy as np 
import pandas as pd 
import requests
import xlsxwriter 
import math 

with open("iex_cloud_api_token.txt", "r") as file:
    IEX_CLOUD_API_TOKEN = file.read()

stocks = pd.read_csv('sp_500_stocks.csv')

symbols_list = stocks['Symbol'].tolist()


def createURL(symbols_list, date, token):
    symbols_joined = ','.join(symbols_list)    
    #"https://sandbox.iexapis.com/stable/stock/market/batch?&symbols=aapl,fb&types=chart&exactDate=20190108&chartByDay=true&token=Tsk_99e87b221ae443588294573fbfa49b6f"    
    batch = 'https://sandbox.iexapis.com/stable/stock/market/batch?'
    symbols = f'&symbols={symbols_joined}'
    types = '&types=chart'
    exactDate = f'&exactDate={date}'
    chartByDay = '&chartByDay=true'
    chartCloseOnly = '&chartCloseOnly=true'
    token = f'&token={token}'    
    return (batch + symbols + types + exactDate + chartByDay + chartCloseOnly 
            + token)

#print(createURL(symbols_list, date, IEX_CLOUD_API_TOKEN))    

df = pd.DataFrame(columns = symbols_list)
df.index.name = 'Date'


for date in range(20190108, 20190111):
    url = createURL(symbols_list, date, IEX_CLOUD_API_TOKEN)
    print(url)
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f'HTTP Response Status Code: {response.status_code}')
    data_json = requests.get(url).json()
    
    # data['AAPL']['chart'][0]['close']
    # data['AAPL']['chart'][0]['date']
    
    for symbol, chart in data_json.items():
        
        #check date        
        try:
            df.at[date, symbol] = chart['chart'][0]['close'] 
        except:
            pass
        



print(df)
    
# =============================================================================
# 
# def chunks(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]
# 
# 
# symbol_groups = list(chunks(stocks['Ticker'], 100))
# 
# symbol_strings = []
# for i in range(0, len(symbol_groups)):
#     symbol_strings.append(','.join(symbol_groups[i]))
# #     print(symbol_strings[i])
# 
# my_columns = ['Ticker', 'Price','Market Capitalization', 'Number Of Shares to Buy']
# final_dataframe = pd.DataFrame(columns = my_columns)
# 
# for symbol_string in symbol_strings:
# #     print(symbol_strings)
#     batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
#     data = requests.get(batch_api_call_url).json()
#     for symbol in symbol_string.split(','):
#         final_dataframe = final_dataframe.append(
#                                         pd.Series([symbol, 
#                                                    data[symbol]['quote']['latestPrice'], 
#                                                    data[symbol]['quote']['marketCap'], 
#                                                    'N/A'], 
#                                                   index = my_columns), 
#                                         ignore_index = True)
# =============================================================================
        
    
# =============================================================================
# print(final_dataframe)
# =============================================================================










