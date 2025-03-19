import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Konfigurera loggning
logging.basicConfig(filename="benchmark_comparison.log", level=logging.INFO)


def calculate_cumulative_returns(price_series):
    """
    Beräknar kumulativ avkastning baserat på prisrörelser.
    """
    try:
        cumulative_returns = (1 + price_series.pct_change()).cumprod()
        logging.info("✅ Kumulativ avkastning beräknad.")
        return cumulative_returns
    except Exception as e:
        logging.error(f"❌ Fel vid beräkning av kumulativ avkastning: {str(e)}")
        return None


def plot_benchmark_comparison(
    strategy_returns,
    benchmark_returns,
    benchmark_name="S&P 500",
    output_file="benchmark_comparison.png",
):
    """
    Jämför AI-strategins prestanda med ett benchmarkindex.
    """
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(strategy_returns, label="AI-Strategi", color="blue")
        plt.plot(
            benchmark_returns, label=f"{benchmark_name}", color="red", linestyle="--"
        )
        plt.axhline(y=1, color="gray", linestyle="--", label="Startvärde")
        plt.legend()
        plt.title(f"Strategins Avkastning vs {benchmark_name}")
        plt.xlabel("Tidsperiod")
        plt.ylabel("Kumulativ Avkastning")
        plt.savefig(output_file)
        plt.close()
        logging.info("✅ Benchmark-jämförelse genererad.")
    except Exception as e:
        logging.error(f"❌ Fel vid generering av benchmark-jämförelse: {str(e)}")


# Exempelanrop
if __name__ == "__main__":
    # Simulerad strategiavkastning och benchmarkavkastning
    np.random.seed(42)
    strategy_prices = pd.Series(np.cumsum(np.random.randn(100)) + 100)
    benchmark_prices = pd.Series(np.cumsum(np.random.randn(100)) + 100)

    strategy_returns = calculate_cumulative_returns(strategy_prices)
    benchmark_returns = calculate_cumulative_returns(benchmark_prices)

    plot_benchmark_comparison(strategy_returns, benchmark_returns)
    print("📊 Benchmark-jämförelse skapad: benchmark_comparison.png")
