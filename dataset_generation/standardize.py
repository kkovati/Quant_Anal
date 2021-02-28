import numpy as np
import pandas as pd

from database_connection import HSMDataset


def standardize(dataframe):
    assert type(dataframe) is pd.DataFrame
    assert len(dataframe.columns) == 4

    concat_array = np.empty((0,))

    # Also checks correct columns
    for column in ['Open', 'High', 'Low', 'Close']:
        temp = dataframe[column].to_numpy()
        concat_array = np.concatenate((concat_array, temp))

    dataframe -= np.mean(concat_array)
    dataframe /= np.std(concat_array)

    return dataframe


if __name__ == '__main__':
    ds = HSMDataset(debug=True)
    df = ds.open_file('ge.us.txt')
    df = df[['Open', 'High', 'Low', 'Close']]
    df = df.iloc[14000:14050]

    import plotly.express as px

    fig = px.scatter(df).update_traces(mode='lines+markers').update_layout(title='Test1', hovermode="x unified").show()

    df_st = standardize(df)

    fig = px.scatter(df_st).update_traces(mode='lines+markers').update_layout(title='Test1',
                                                                              hovermode="x unified").show()
