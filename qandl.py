# https://www.quandl.com/tools/python
# registered to this site
# API key:HNG5Ky6fcw6SsTQZuxNS

# https://www.youtube.com/watch?v=EYnC4ACIt2g&t=373s

import quandl
import matplotlib.pyplot as plt

# quandl.ApiConfig.api_key = 'HNG5Ky6fcw6SsTQZuxNS'
# Get the data for Coca-cola
df = quandl.get("WIKI/KO", start_date="2016-01-01", end_date="2018-01-01", api_key="HNG5Ky6fcw6SsTQZuxNS")

print(df.head())

# Plot the prices
df.Close.plot()
plt.show()
