import logging

import numpy as np

# Konfigurera loggning
logging.basicConfig(filename="monte_carlo_simulation.log", level=logging.INFO)


def monte_carlo_simulation(
    initial_value, mean_return, volatility, days=252, simulations=1000
):
    """
    Simulerar portföljens framtida värde med Monte Carlo-metoden.
    """
    try:
        results = []
        for _ in range(simulations):
            daily_returns = np.random.normal(
                mean_return / days, volatility / np.sqrt(days), days
            )
            price_series = initial_value * (1 + daily_returns).cumprod()
            results.append(price_series[-1])

        expected_value = np.mean(results)
        logging.info("✅ Monte Carlo-simulering genomförd.")
        return expected_value
    except Exception as e:
        logging.error(f"❌ Fel vid Monte Carlo-simulering: {str(e)}")
        return None


# Exempelanrop
if __name__ == "__main__":
    expected_value = monte_carlo_simulation(
        initial_value=100000, mean_return=0.07, volatility=0.2
    )
    print(f"📢 Förväntat framtida portföljvärde: {expected_value:.2f}")
