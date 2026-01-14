from research import ResearchEngine
from visualize import ResultVisualizer


def main():
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
    

if __name__ == "__main__":
    main()



