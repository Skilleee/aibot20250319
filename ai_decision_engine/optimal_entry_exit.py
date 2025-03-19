import logging

import numpy as np
import pandas as pd

# Konfigurera loggning
logging.basicConfig(filename="optimal_entry_exit.log", level=logging.INFO)


def moving_average(prices, window=20):
    """
    Beräknar glidande medelvärde för en aktie.
    """
    try:
        ma = prices.rolling(window=window).mean()
        logging.info(f"✅ Glidande medelvärde ({window} dagar) beräknat.")
        return ma
    except Exception as e:
        logging.error(f"❌ Fel vid beräkning av glidande medelvärde: {str(e)}")
        return None


def bollinger_bands(prices, window=20, num_std=2):
    """
    Beräknar Bollinger Bands för att identifiera köp- och säljnivåer.
    """
    try:
        sma = moving_average(prices, window)
        std_dev = prices.rolling(window=window).std()
        upper_band = sma + (std_dev * num_std)
        lower_band = sma - (std_dev * num_std)
        logging.info(
            f"✅ Bollinger Bands beräknat med {window}-dagars fönster och {num_std} standardavvikelser."
        )
        return upper_band, lower_band
    except Exception as e:
        logging.error(f"❌ Fel vid beräkning av Bollinger Bands: {str(e)}")
        return None, None


def optimal_entry_exit_strategy(prices, window=20, num_std=2):
    """
    Genererar köp- och säljnivåer baserat på Bollinger Bands.
    """
    try:
        upper_band, lower_band = bollinger_bands(prices, window, num_std)
        signals = pd.Series(index=prices.index, dtype="object")
        signals[prices < lower_band] = "BUY"
        signals[prices > upper_band] = "SELL"
        logging.info("✅ Optimal entry/exit-strategi beräknad.")
        return signals
    except Exception as e:
        logging.error(f"❌ Fel vid generering av entry/exit-strategi: {str(e)}")
        return None


# Exempelanrop
if __name__ == "__main__":
    # Simulerad prisdata
    np.random.seed(42)
    simulated_prices = pd.Series(np.cumsum(np.random.randn(100)) + 100)

    ma_20 = moving_average(simulated_prices)
    print(f"📈 20-dagars glidande medelvärde:\n{ma_20.tail()}")

    upper_band, lower_band = bollinger_bands(simulated_prices)
    print(
        f"📊 Bollinger Bands:\nÖvre band:\n{upper_band.tail()}\nNedre band:\n{lower_band.tail()}"
    )

    signals = optimal_entry_exit_strategy(simulated_prices)
    print(f"📢 Entry/Exit-signaler:\n{signals.tail()}")
