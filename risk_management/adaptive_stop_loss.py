import logging

import numpy as np
import pandas as pd

# Konfigurera loggning
logging.basicConfig(filename="adaptive_stop_loss.log", level=logging.INFO)


def calculate_atr(price_data, window=14):
    """
    Beräknar Average True Range (ATR) för att mäta volatilitet.
    """
    try:
        high_low = price_data["High"] - price_data["Low"]
        high_close = np.abs(price_data["High"] - price_data["Close"].shift(1))
        low_close = np.abs(price_data["Low"] - price_data["Close"].shift(1))
        true_range = pd.DataFrame(
            {"TR1": high_low, "TR2": high_close, "TR3": low_close}
        ).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        logging.info(f"✅ ATR beräknad över {window} perioder.")
        return atr
    except Exception as e:
        logging.error(f"❌ Fel vid beräkning av ATR: {str(e)}")
        return None


def adaptive_stop_loss(price_data, atr_multiplier=3, window=14):
    """
    Beräknar en dynamisk stop-loss baserat på ATR.
    """
    try:
        atr = calculate_atr(price_data, window)
        stop_loss = price_data["Close"] - (atr_multiplier * atr)
        logging.info("✅ Adaptiv stop-loss beräknad.")
        return stop_loss
    except Exception as e:
        logging.error(f"❌ Fel vid beräkning av adaptiv stop-loss: {str(e)}")
        return None


# Exempelanrop
if __name__ == "__main__":
    # Simulerad prisdata
    np.random.seed(42)
    price_data = pd.DataFrame(
        {
            "High": np.random.uniform(100, 110, 100),
            "Low": np.random.uniform(90, 100, 100),
            "Close": np.random.uniform(95, 105, 100),
        }
    )

    atr = calculate_atr(price_data)
    print(f"📊 ATR: {atr.tail()}")

    stop_loss = adaptive_stop_loss(price_data)
    print(f"🚨 Adaptiv Stop-Loss: {stop_loss.tail()}")
