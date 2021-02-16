import pandas as pd
import requests

from single_symbol_max_history import single_symbol_max_history


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
    stocks = pd.read_csv('../data/sp_10_stocks.csv')
    symbols_list = stocks['Symbol'].tolist()

    dataframe = pd.DataFrame(columns=symbols_list)
    dataframe.index.name = 'Date'

    for i, symbol in enumerate(symbols_list):
        print(f'Request {symbol} {i + 1}/{len(symbols_list)}')

        new_df = single_symbol_max_history(symbol)

        dataframe = pd.concat([dataframe, new_df], axis=1)

    dataframe.sort_index(inplace=True)

    dataframe.to_csv(path_or_buf='..\\data\\csv\\temp.csv')

    return dataframe


if __name__ == '__main__':
    df = query_history()

    print(df)
