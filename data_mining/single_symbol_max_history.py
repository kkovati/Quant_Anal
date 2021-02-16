import pandas as pd
from get_request import get_request


def single_symbol_max_history(symbol):
    """
    Queries all available close price of a single symbol
    """
    with open('..\\token\\iex_cloud_api_token.txt', 'r') as file:
        token = file.read()

    print(f'Request max history of {symbol}')

    # https://sandbox.iexapis.com/stable/stock/AAPL/chart/max?chartCloseOnly=true&token=Tpk_450fc249b778447392ad16144bab7ee1
    # https://sandbox.iexapis.com/stable/stock/AAPL/chart/max?&token=Tpk_450fc249b778447392ad16144bab7ee1
    url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/chart/max?chartCloseOnly=true&token={token}'
    # url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/chart/max?&token={token}'

    data_json = get_request(url)

    dataframe = pd.DataFrame(columns=[symbol])
    dataframe.index.name = 'Date'

    for daily_data in data_json:
        # date = int(daily_data['date'].replace('-', ''))
        date = daily_data['date']
        dataframe.at[date, symbol] = daily_data['close']

    dataframe.sort_index(inplace=True)

    return dataframe


if __name__ == '__main__':
    df = single_symbol_max_history(symbol='AAPL')

    print(df)

    assert df.columns[0] == 'AAPL'
    assert len(df.columns) == 1
    assert df.index.name == "Date"
    # assert df.index[0] > 20060000
    assert len(df.index) > 3500
    assert len(df) > 3500

    print("Test1 OK")


# HISTORICAL_CLOSE_PRICES		477,072			2			70
# HISTORICAL_PRICES			    106,830			10			3
#
# Tpk_450fc249b778447392ad16144bab7ee1		583,902
#
#
#
# ----After I downloaded full historical data of 10 stocks:
#
# HISTORICAL_CLOSE_PRICES		477,072			2			70
# HISTORICAL_PRICES			    714,890			10			23
#
#
# Tpk_450fc249b778447392ad16144bab7ee1	1,191,962  		delta: 608,060
#
#
# ----After I downloaded only close historical data of 10 stocks:
#
#
# HISTORICAL_CLOSE_PRICES		598,684			2			90
# HISTORICAL_PRICES			    714,890			10			23
#
#
#
# Tpk_450fc249b778447392ad16144bab7ee1	1,313,574   	delta: 121,612