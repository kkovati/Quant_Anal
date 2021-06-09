from Zipline.api import order, record, symbol
from Zipline.algorithm import TradingAlgorithm

# https://blog.quantinsti.com/introduction-zipline-python/

def initialize(context):
   pass

def handle_data(context, data):
  order(symbol('AAPL'), 10)
  record(AAPL=data.current(symbol('AAPL'), 'price'))

algo_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data)
perf_manual = algo_obj.run(data)