import yfinance as yf
import pandas as pd

class DataHandler:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.prices = None
        self.returns = None

    def load_data(self):
        self.prices = yf.download(
            tickers= self.tickers,
            start= self.start_date, 
            end= self.end_date,
            auto_adjust=True
        )[['Close']]

        self.prices = self.prices['Close']
        return self.prices

    def compute_returns(self):
        self.returns = self.prices.pct_change().fillna(0)
        return self.returns
