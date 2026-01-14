import yfinance as yf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

"""Assumptions:
daily close prices
you trade at close 
no leverage
"""

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
        



class ResearchEngine:
    def __init__(
        self,
        tickers,
        start_date,
        end_date,
        ma_periods,
        weights,
        cost_rate,
        initial_capital,
        train_split=0.7,
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.ma_periods = ma_periods
        self.weights = weights
        self.cost_rate = cost_rate
        self.initial_capital = initial_capital
        self.train_split = train_split

        # Outputs / artifacts
        self.split_date = None
        self.best_ma = None

        self.results_train = None
        self.results_test = None

        self.equity_full = None   # DataFrame: columns ['best', 'buy_and_hold']
        self.equity_test = None   # DataFrame: columns ['best', 'buy_and_hold']

    @staticmethod
    def _split_by_ratio(df, ratio):
        """Split a DataFrame/Series by row count ratio using iloc (safe for DateTimeIndex)."""
        n = len(df)
        cut = int(ratio * n)
        return df.iloc[:cut], df.iloc[cut:]

    @staticmethod
    def _compute_drawdown(equity: pd.Series) -> pd.Series:
        peak = equity.cummax()
        return equity / peak - 1

    @staticmethod
    def _run_strategy(prices, returns, strategy, weights, cost_rate, initial_capital):
        """Run one strategy end-to-end and return (equity, returns_portfolio, metrics_dict)."""
        signals_df = strategy.generate_signals(prices)

        portfolio = Portfolio(weights)
        equity = portfolio.run(signals_df, returns, cost_rate, initial_capital)
        returns_portfolio = portfolio.returns_portfolio

        analyzer = PerformanceAnalyzer(equity, returns_portfolio)
        metrics = analyzer.run()  # dict of scalar metrics
        return equity, returns_portfolio, metrics

    def perform_backtests(self):
        # 1) Load data
        data = DataHandler(self.tickers, self.start_date, self.end_date)
        prices = data.load_data()
        returns = data.compute_returns()

        # 2) Split
        prices_train, prices_test = self._split_by_ratio(prices, self.train_split)
        returns_train, returns_test = self._split_by_ratio(returns, self.train_split)
        self.split_date = prices_test.index[0] if len(prices_test) > 0 else prices.index[-1]

        # 3) TRAIN sweep -> pick best_ma
        train_rows = []
        for ma in self.ma_periods:
            equity, r_port, metrics = self._run_strategy(
                prices_train, returns_train, StrategyMa(ma),
                self.weights, self.cost_rate, self.initial_capital
            )
            train_rows.append({**{"dataset": "train", "strategy": "MA", "ma_period": ma}, **metrics})

        self.results_train = pd.DataFrame(train_rows)

        # Choose best on TRAIN (exclude any non-MA rows; here all are MA anyway)
        best_idx = self.results_train["sharpe_annual"].idxmax()
        self.best_ma = int(self.results_train.loc[best_idx, "ma_period"])

        # 4) TEST evaluation for best_ma and buy&hold
        test_rows = []

        equity_best_test, r_best_test, metrics_best_test = self._run_strategy(
            prices_test, returns_test, StrategyMa(self.best_ma),
            self.weights, self.cost_rate, self.initial_capital
        )
        test_rows.append({**{"dataset": "test", "strategy": "MA", "ma_period": self.best_ma}, **metrics_best_test})

        equity_bh_test, r_bh_test, metrics_bh_test = self._run_strategy(
            prices_test, returns_test, StrategyBuyHold(),
            self.weights, self.cost_rate, self.initial_capital
        )
        test_rows.append({**{"dataset": "test", "strategy": "buy_and_hold", "ma_period": None}, **metrics_bh_test})

        self.results_test = pd.DataFrame(test_rows)

        self.equity_test = pd.DataFrame(
            {"best": equity_best_test, "buy_and_hold": equity_bh_test}
        )

        # 5) FULL-period curves for best_ma and buy&hold (for “whole period” plots)
        equity_best_full, r_best_full, metrics_best_full = self._run_strategy(
            prices, returns, StrategyMa(self.best_ma),
            self.weights, self.cost_rate, self.initial_capital
        )
        equity_bh_full, r_bh_full, metrics_bh_full = self._run_strategy(
            prices, returns, StrategyBuyHold(),
            self.weights, self.cost_rate, self.initial_capital
        )

        self.equity_full = pd.DataFrame(
            {"best": equity_best_full, "buy_and_hold": equity_bh_full}
        )

        # Return a compact “bundle” (nice for your Visualizer)
        return {
            "best_ma": self.best_ma,
            "split_date": self.split_date,
            "results_train": self.results_train,
            "results_test": self.results_test,
            "equity_full": self.equity_full,
            "equity_test": self.equity_test,
        }



class ResultVisualizer:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.best_ma = bundle["best_ma"]
        self.split_date = bundle["split_date"]

    @staticmethod
    def _drawdown(equity: pd.Series) -> pd.Series:
        return equity / equity.cummax() - 1

    def plot_full_equity_and_drawdown(self):
        equity = self.bundle["equity_full"]
        dd_best = self._drawdown(equity["best"])
        dd_bh = self._drawdown(equity["buy_and_hold"])

        # Equity
        plt.figure()
        plt.plot(equity.index, equity["best"], label=f"Best MA (train-selected): {self.best_ma}")
        plt.plot(equity.index, equity["buy_and_hold"], label="Buy & Hold")
        plt.axvline(self.split_date, linestyle="--", linewidth=1, label="Train/Test split")
        plt.title("Equity Curve (Full Period)")
        plt.legend()
        

        # Drawdown
        plt.figure()
        plt.plot(dd_best.index, dd_best, label="DD Best MA")
        plt.plot(dd_bh.index, dd_bh, label="DD Buy & Hold")
        plt.axvline(self.split_date, linestyle="--", linewidth=1, label="Train/Test split")
        plt.axhline(0, linewidth=1)
        plt.title("Drawdown (Full Period)")
        plt.legend()
        

    def plot_test_equity_and_drawdown(self):
        equity = self.bundle["equity_test"]
        dd_best = self._drawdown(equity["best"])
        dd_bh = self._drawdown(equity["buy_and_hold"])

        plt.figure()
        plt.plot(equity.index, equity["best"], label=f"Best MA on TEST: {self.best_ma}")
        plt.plot(equity.index, equity["buy_and_hold"], label="Buy & Hold on TEST")
        plt.title("Equity Curve (Test Only)")
        plt.legend()
        

        plt.figure()
        plt.plot(dd_best.index, dd_best, label="DD Best MA (Test)")
        plt.plot(dd_bh.index, dd_bh, label="DD Buy & Hold (Test)")
        plt.axhline(0, linewidth=1)
        plt.title("Drawdown (Test Only)")
        plt.legend()
        plt.show()



#  Run example 
start_date = '2020-10-01'
end_date = '2026-01-01'

tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD']
ma_periods = [2, 4, 5, 6, 7, 10, 15, 20, 27, 35, 40, 60, 80, 110, 130, 150, 200, 252]

cost_rate = 0.001
initial_capital = 100
weights = [0.25, 0.25, 0.25, 0.25]

train_split = 0.7  # 70% train, 30% test

bt1 = ResearchEngine(
    tickers, start_date, end_date,
    ma_periods, weights, cost_rate, initial_capital,
    train_split=train_split
)

bundle = bt1.perform_backtests()

print("Best MA chosen on TRAIN:", bundle["best_ma"])
print("\nTRAIN results:")
print(bundle["results_train"].sort_values("sharpe_annual", ascending=False))
print("\nTEST results:")
print(bundle["results_test"].sort_values("sharpe_annual", ascending=False))

viz = ResultVisualizer(bundle)
viz.plot_full_equity_and_drawdown()
viz.plot_test_equity_and_drawdown()
 