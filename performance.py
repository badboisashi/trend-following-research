import pandas as pd 
import numpy as np

class PerformanceAnalyzer:
    def __init__(self, portfolio_equity,returns):
        self.portfolio_equity = portfolio_equity
        self.returns = returns
        self.risk_metrics = None
        self.vol_annual = None
        self.return_total = None
        self.return_annual = None
        self.sharpe_annual =None
        self.max_dd = None
        self.rolling_metrics = None
        

    def compute_vol_annual(self):
        self.vol_annual = self.returns.std() * np.sqrt(252)
        return self.vol_annual
    
    def compute_return_total(self):
        self.return_total = self.portfolio_equity.iloc[-1] / self.portfolio_equity.iloc[0] - 1
        return self.return_total
    
    def compute_return_annual(self):
        self.return_annual = (self.return_total + 1) ** (252 / len(self.portfolio_equity)) - 1
        return self.return_annual
    
    def compute_sharpe_annual(self):
        # assumes rf = 0
        self.sharpe_annual = self.return_annual / self.vol_annual
        return self.sharpe_annual
    
    def compute_max_dd(self):
        drawdowns = self.portfolio_equity / self.portfolio_equity.cummax() - 1
        self.max_dd = drawdowns.min()
        return self.max_dd
    
    def compute_rolling_metrics(self):
        rolling_volatility = self.returns.rolling(63).std()

        cummulative_peaks = self.portfolio_equity.cummax()
        drawdown_curve = (self.portfolio_equity - cummulative_peaks) / cummulative_peaks

        rolling_sharpe = self.returns.rolling(63).mean() / rolling_volatility * np.sqrt(252)

        self.rolling_metrics = pd.concat(
            [
                rolling_volatility.rename('rolling_volatility_63'),
                drawdown_curve.rename('drawdown_curve'), 
                rolling_sharpe.rename('rolling_sharpe_63')
                ],
                axis=1
            )

        return self.rolling_metrics 
    
    
    def get_risk_metrics(self):
        self.risk_metrics = {'volatility_annual':self.vol_annual,
        'return_total':self.return_total,
        'return_annual':self.return_annual,
        'sharpe_annual':self.sharpe_annual,
        'max_dd':self.max_dd}
        return self.risk_metrics

    def run(self):
        """
        assumes rf = 0
        """
        self.compute_vol_annual()
        self.compute_return_total()
        self.compute_return_annual()
        self.compute_sharpe_annual()
        self.compute_max_dd()
        self.get_risk_metrics()
        self.compute_rolling_metrics()
        return self.risk_metrics
