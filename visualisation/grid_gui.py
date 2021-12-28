import time
import tkinter as tk
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def sample_1():
    data1 = {'Country': ['US', 'CA', 'GER', 'UK', 'FR'],
             'GDP_Per_Capita': [45000, 42000, 52000, 49000, 47000]
             }
    df1 = DataFrame(data1, columns=['Country', 'GDP_Per_Capita'])

    data2 = {'Year': [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010],
             'Unemployment_Rate': [9.8, 12, 8, 7.2, 6.9, 7, 6.5, 6.2, 5.5, 6.3]
             }
    df2 = DataFrame(data2, columns=['Year', 'Unemployment_Rate'])

    data3 = {'Interest_Rate': [5, 5.5, 6, 5.5, 5.25, 6.5, 7, 8, 7.5, 8.5],
             'Stock_Index_Price': [1500, 1520, 1525, 1523, 1515, 1540, 1545, 1560, 1555, 1565]
             }
    df3 = DataFrame(data3, columns=['Interest_Rate', 'Stock_Index_Price'])

    root = tk.Tk()

    figure1 = plt.Figure(figsize=(6, 5), dpi=100)
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, root)
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    df1 = df1[['Country', 'GDP_Per_Capita']].groupby('Country').sum()
    df1.plot(kind='bar', legend=True, ax=ax1)
    ax1.set_title('Country Vs. GDP Per Capita')

    figure2 = plt.Figure(figsize=(5, 4), dpi=100)
    ax2 = figure2.add_subplot(111)
    line2 = FigureCanvasTkAgg(figure2, root)
    line2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    df2 = df2[['Year', 'Unemployment_Rate']].groupby('Year').sum()
    df2.plot(kind='line', legend=True, ax=ax2, color='r', marker='o', fontsize=10)
    ax2.set_title('Year Vs. Unemployment Rate')

    figure3 = plt.Figure(figsize=(5, 4), dpi=100)
    ax3 = figure3.add_subplot(111)
    ax3.scatter(df3['Interest_Rate'], df3['Stock_Index_Price'], color='g')
    scatter3 = FigureCanvasTkAgg(figure3, root)
    scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    ax3.legend(['Stock_Index_Price'])
    ax3.set_xlabel('Interest Rate')
    ax3.set_title('Interest Rate Vs. Stock Index Price')

    root.mainloop()


class GUI:
    def __init__(self):
        self.win = tk.Tk()
        self.prices = [0] * 100

    def update(self, price, wallet, status):
        self.prices.pop(0)
        self.prices.append(price)

        figure2 = plt.Figure(figsize=(5, 4), dpi=100)
        ax2 = figure2.add_subplot(111)
        line2 = FigureCanvasTkAgg(figure2, self.win)
        line2.get_tk_widget().grid(row=0, columnspan=100)
        ax2.plot(self.prices)
        ax2.set_title('Title')

        status_label_pointer = tk.Label(text="Status:")
        status_label_content = tk.Label(text="in trade")
        wallet_label_pointer = tk.Label(text="Wallet:")
        wallet_label_content = tk.Label(text=wallet)

        status_label_pointer.grid(row=1, column=0)
        status_label_content.grid(row=1, column=1)
        wallet_label_pointer.grid(row=2, column=0)
        wallet_label_content.grid(row=2, column=1)

        self.win.update_idletasks()
        self.win.update()


def main():
    import random
    
    gui = GUI()

    while True:
        gui.update(random.random(), 100, "asd")
        time.sleep(1)


if __name__ == '__main__':
    main()
