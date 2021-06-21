import numpy as np
import pandas as pd

from dataset_generation.hsm_dataset import HSMDataset


def standardize(interval):
    if type(interval) is pd.DataFrame:
        return standardize_ohlc_dataframe(interval)
    elif type(interval) is pd.Series:
        return standardize_series(interval)
    elif type(interval) is np.ndarray:
        return standardize_ndarray(interval)
    else:
        raise Exception('Wrong input type')


def standardize_ohlc_dataframe(dataframe):
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


def standardize_series(series):
    assert type(series) is pd.Series

    series -= np.mean(series)
    series /= np.std(series)

    return series


def standardize_ndarray(array):
    assert isinstance(array, np.ndarray)
    assert array.shape[0] == 4

    array = array.copy()  # Copy input to prevent side effect
    array -= np.mean(array)
    if np.std(array) > 0:
        array /= np.std(array)
    else:
        print('\nWarning: np.std(array) < 0 @ standardize_ndarray()')
        print(array)
    return array


if __name__ == '__main__':
    import plotly.express as px

    ds = HSMDataset(test_size=0.1, debug=True)
    df = ds.open_file('ge.us.txt')
    df = df[['Open', 'High', 'Low', 'Close']]
    df = df.iloc[14000:14050]

    # Plot dataframe
    fig = px.scatter(df).update_traces(mode='lines+markers').update_layout(title='Dataframe normal',
                                                                           hovermode="x unified").show()

    # Plot standardized dataframe
    df_st = standardize(df.copy())
    fig = px.scatter(df_st).update_traces(mode='lines+markers').update_layout(title='Dataframe standardize',
                                                                              hovermode="x unified").show()

    # Plot series
    ser = df['Close']
    fig = px.scatter(ser).update_traces(mode='lines+markers').update_layout(title='Series normal',
                                                                            hovermode="x unified").show()

    # Plot standardized series
    ser_st = standardize(ser.copy())
    fig = px.scatter(ser_st).update_traces(mode='lines+markers').update_layout(title='Series standardize',
                                                                               hovermode="x unified").show()

    # Plot ndarray
    nda = np.zeros((4, 50))
    nda[0] = df['Open']
    nda[1] = df['High']
    nda[2] = df['Low']
    nda[3] = df['Close']
    fig = px.scatter(np.transpose(nda)).update_traces(mode='lines+markers').update_layout(title='Ndarray normal',
                                                                                          hovermode="x unified").show()

    # Plot standardized ndarray
    nda_st = standardize(nda.copy())
    fig = px.scatter(np.transpose(nda_st)).update_traces(mode='lines+markers')
    fig.update_layout(title='Ndarray normal', hovermode="x unified").show()
