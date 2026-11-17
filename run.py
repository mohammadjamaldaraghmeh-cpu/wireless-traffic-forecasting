"""Run the full experiment: load data, forecast with every model, report errors and
energy-saving results, save a table and one plot per cell into results/.

    python run.py                          # synthetic data
    python run.py --milan data/milan       # real Telecom Italia data
    python run.py --milan data/milan --cells 5060 4259 4456
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_milan, make_synthetic, train_test_split
from src.evaluate import metrics, sleep_policy
from src.models import hybrid_forecast, lstm_forecast, sarima_forecast, seasonal_naive


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--milan", help="folder with sms-call-internet-mi-*.txt files")
    p.add_argument("--cells", type=int, nargs="*", default=[5060, 4259, 4456],
                   help="Milan grid cell ids to use")
    p.add_argument("--test-days", type=int, default=7)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--sleep-quantile", type=float, default=0.2,
                   help="a cell may sleep when load is below this quantile of its training load")
    args = p.parse_args()

    if args.milan:
        df = load_milan(args.milan, cells=args.cells)
        source = "Milan"
    else:
        df = make_synthetic()
        source = "synthetic"
    print(f"Data: {source}, {df.shape[1]} cells, {len(df)} hours "
          f"({df.index[0]} -> {df.index[-1]})")

    train_df, test_df = train_test_split(df, args.test_days)
    out = Path("results")
    out.mkdir(exist_ok=True)

    rows = []
    for cell in df.columns:
        train, test = train_df[cell], test_df[cell]
        actual = test.to_numpy()
        print(f"  {cell}: fitting models...")

        sarima_test, sarima_train = sarima_forecast(train, test)
        preds = {
            "Naive (yesterday)": seasonal_naive(train, test),
            "SARIMA": sarima_test,
            "LSTM": lstm_forecast(train, test, epochs=args.epochs),
            "Hybrid SARIMA+LSTM": hybrid_forecast(
                train, test, sarima_test, sarima_train, epochs=args.epochs),
        }

        threshold = float(train.quantile(args.sleep_quantile))
        for name, pred in preds.items():
            rows.append({"cell": cell, "model": name,
                         **metrics(actual, pred),
                         **sleep_policy(actual, pred, threshold)})

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(test.index, actual, color="black", linewidth=2, label="Actual")
        for name, pred in preds.items():
            ax.plot(test.index, pred, linewidth=1, alpha=0.8, label=name)
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label="Sleep threshold")
        ax.set_title(f"{cell} - next-hour forecast ({source} data)")
        ax.set_ylabel("Traffic load")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"forecast_{cell}.png", dpi=120)
        plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(out / "per_cell.csv", index=False)
    summary = (table.drop(columns="cell").groupby("model").mean()
               .sort_values("RMSE").round(2))
    summary.to_csv(out / "summary.csv")
    print("\nAverage over cells (sorted by RMSE):")
    print(summary.to_string())
    print(f"\nSaved tables and plots to {out.resolve()}")


if __name__ == "__main__":
    main()
