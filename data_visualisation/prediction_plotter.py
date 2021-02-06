import pandas as pd
import plotly.graph_objects as go


# https://plotly.com/python/peak-finding/
# https://plotly.com/python/marker-style/
# https://plotly.com/python/hover-text-and-formatting/

def plot_df(df):
    assert len(df.columns) == 3
    symbol = df.columns[0]

    prices = go.Scatter(x=df.index, y=df[symbol], name='Price')

    buys = go.Scatter(x=df.index, y=df["Buy"], name='Buy', mode='markers',
                      marker=dict(size=10, color='orange', symbol='arrow-up'))

    predictions = go.Scatter(x=df.index, y=df["Prediction"], name='Prediction', mode='markers',
                             marker=dict(size=8, color='red', symbol='cross'))

    fig = go.Figure()
    fig.add_trace(prices)
    fig.add_trace(buys)
    fig.add_trace(predictions)

    fig.update_layout(title=f'{symbol} Prices over time (2014)',
                      xaxis_title="Date",
                      yaxis_title="Price",
                      hovermode="x unified")

    fig.show()

    # from plotly.offline import plot
    # plot(fig)


if __name__ == '__main__':
    # Test 1
    df = pd.read_csv("test_data/AAPL_2014_test.csv").set_index("Date")

    plot_df(df)
