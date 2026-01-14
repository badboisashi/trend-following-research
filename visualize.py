import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
