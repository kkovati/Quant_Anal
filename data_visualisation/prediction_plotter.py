import pandas as pd
from plotly.graph_objects import Scatter, Figure
from plotly.offline import plot


def plot_scat(prices, date_index_buy, date_index_sell, sell_price, profit, sl):
    
    scatter = Scatter(x=prices.index, y=prices.values)
    figure = Figure([scatter])
    
    color = 'Green'
    # vertical line
    figure.add_shape(type="line",
                     x0=prices.index[date_index_buy], # buy date
                     y0=prices.values[date_index_buy], # buy price
                     x1=prices.index[date_index_sell], # sell date
                     y1=sell_price, # sell date
                     line=dict(color='Grey',width=2)) 

    figure.add_vrect(x0=27-0.5, 
                     x1=29+0.5,
                     annotation_text="ideal invest", 
                     annotation_position="bottom",
                     fillcolor="Red", 
                     opacity=0.5,
                     layer="below", 
                     line_width=1)
    
    figure.add_vrect(x0=28-0.5, 
                     x1=30+0.5,
                     annotation_text="proposed invest", 
                     annotation_position="bottom",
                     fillcolor="Green", 
                     opacity=0.5,
                     layer="below", 
                     line_width=1)
    
    plot(figure)
   
        
def init_figure(self, analyzed_interval, predictions, model):
    candlestick = Candlestick(x=analyzed_interval[:,0], 
                              open=analyzed_interval[:,1], 
                              high=analyzed_interval[:,2], 
                              low=analyzed_interval[:,3], 
                              close=analyzed_interval[:,4])
    
    figure = Figure(data=[candlestick])
    
    model.categories.sort()
    steps = ', '.join(['+' + str(c) + '%' for c in model.categories])
    title = ('Predictions of ' + model.name + ' for the next ' + 
             str(model.prediction_interval) + 
             ' days based on the previous ' + str(model.input_interval) + 
             ' days. Steps: ' + steps)
    figure.update_layout(title=title, yaxis_title='Price')
    
    return figure        
        
        
def visualize_predictions(self, figure, analyzed_interval, predictions, model):
   
    prediction_interval = model.prediction_interval
    
    for i, prediction in enumerate(predictions):            
        # do not draw those predictions which cannot be verified
        if i >= len(analyzed_interval) - prediction_interval:
            return
        
        if prediction == 0:
            ratio = model.categories[0] / 100 + 1
            text = '< ' + str(model.categories[0]) + '%' 
            color = 'Red'
        else:                
            ratio = model.categories[prediction-1] / 100 + 1
            text = '> ' + str(model.categories[prediction-1]) + '%'
            color = 'Green'
        
        # vertical line
        figure.add_shape(type="line",
                         x0=analyzed_interval[i][0], # date
                         y0=analyzed_interval[i][4], # close price
                         x1=analyzed_interval[i][0],
                         y1=(float)(analyzed_interval[i][4]) * ratio,
                         line=dict(color=color,width=2, dash="dot"))
        
        # horizontal line
        figure.add_shape(type="line",
                         x0=analyzed_interval[i][0],
                         y0=(float)(analyzed_interval[i][4]) * ratio,
                         x1=analyzed_interval[i + prediction_interval][0],
                         y1=(float)(analyzed_interval[i][4]) * ratio,
                         line=dict(color=color,width=2, dash="dot"))
        
        figure.add_annotation(x=analyzed_interval[i][0],
                              y=(float)(analyzed_interval[i][4]) * ratio,
                              text=text)
        
 
if __name__ == '__main__':
    
    # Test 1
    data = [136,134,133,130,135,140,142,140,135]
    index = [24,25,26,27,28,29,30,31,32]
    prices = pd.Series(data=data, index=index, dtype='float64', name='AAPL')
    print(prices.values)    

    plot_scat(prices, date_index_buy=2, date_index_sell=7, sell_price=139, 
              profit=0, sl=0)
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   