import yfinance as yf
import matplotlib.pyplot as plt

# Get the data for the stock Apple by specifying the stock ticker, start date, and end date
data = yf.download('AAPL', '2016-01-01', '2018-01-01')

# Plot the close prices


data.Close.plot()
plt.show()
