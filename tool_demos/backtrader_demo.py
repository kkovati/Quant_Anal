import backtrader as bt
import backtrader.analyzers as btanalyzers
import matplotlib
from datetime import datetime

# pip install backtrader
# pip install requests
# pip install matplotlib==3.2.2

# https://www.backtrader.com/docu/

class MaCrossStrategy(bt.Strategy):

    def __init__(self):
        ma_fast = bt.ind.SMA(period=10)
        ma_slow = bt.ind.SMA(period=50)

        self.crossover = bt.ind.CrossOver(ma_fast, ma_slow)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()


if __name__ == '__main__':
    cerebro = bt.Cerebro()

    data = bt.feeds.YahooFinanceData(dataname='AAPL', fromdate=datetime(2010, 1, 1), todate=datetime(2020, 1, 1))
    cerebro.adddata(data)

    cerebro.addstrategy(MaCrossStrategy)

    cerebro.broker.setcash(1000000.0)

    cerebro.addsizer(bt.sizers.PercentSizer, percents=10)

    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(btanalyzers.Transactions, _name="trans")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")

    back = cerebro.run()

    cerebro.broker.getvalue()

    back[0].analyzers.sharpe.get_analysis()

    back[0].analyzers.trans.get_analysis()

    back[0].analyzers.trades.get_analysis()

    cerebro.plot()