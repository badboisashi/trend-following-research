import pandas as pd
import numpy as np

class Portfolio:
    def __init__(self, weights):
        self.weights = weights
        self.returns_net = None
        self.returns_portfolio = None
        self.positions = None
        self.trades = None
        self.returns_weighted_net = None
        self.equity = None
        

    def compute_positions(self, signals):
        self.positions = signals.shift(1).fillna(0)
        return self.positions

    def get_trades(self):   
        self.trades = self.positions.diff().fillna(0)       
        return self.trades

    def get_net_returns(self, cost_rate, returns):
        assert self.positions.shape == returns.shape

        self.weights = pd.Series(self.weights, index= returns.columns)
        self.returns_net = returns * self.positions - self.trades.abs() * cost_rate

    def get_returns_weighted_net(self):
        self.returns_weighted_net = self.returns_net.mul(self.weights, axis=1)
        return self.returns_weighted_net

    def compute_portfolio_returns(self):
        self.returns_portfolio = self.returns_weighted_net.sum(axis=1)
        return self.returns_portfolio

    def compute_equity(self, initial_capital):
        self.equity = initial_capital * (1 + self.returns_portfolio).cumprod()
        return self.equity

    def run(self,signals, returns, cost_rate, initial_capital):
        """
        :param signals: DataFrame (dates x assets) comes from StrategyMa Class
        :param returns: DataFrame (dates x assets) comes from DataHandler Class
        :param cost_rate: float
        :param initial_capital: float
        """
        self.compute_positions(signals)
        self.get_trades()
        self.get_net_returns(cost_rate, returns)
        self.get_returns_weighted_net()
        self.compute_portfolio_returns()
        self.compute_equity(initial_capital)
        return self.equity
