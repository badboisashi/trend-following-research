import pandas as pd
from data import DataHandler
from strategy import StrategyBuyHold, StrategyMa
from portfolio import Portfolio
from performance import PerformanceAnalyzer

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
