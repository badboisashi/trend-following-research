import pandas as pd
import numpy as np

class StrategyBuyHold:
    def __init__(self):
        self.signals = None

    def generate_signals(self, prices):
        """
        prices: Dataframe (dates x assets)
        """
        self.signals = pd.DataFrame(1, index=prices.index, columns=prices.columns)
        return self.signals

class StrategyMa:
    def __init__(self, ma_period):
        self.ma_period = ma_period
        self.signals = None

    def generate_signals(self, prices):
        """
        prices: Dataframe (dates x assets)
        returns: Dataframe signals (dates x assets)
        """
        conditions = (prices.rolling(self.ma_period).mean() < prices)
        self.signals = conditions.astype(int)
        
        return self.signals
