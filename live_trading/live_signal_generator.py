import logging

import numpy as np
import pandas as pd

# Konfigurera loggning
logging.basicConfig(filename="live_signal_generator.log", level=logging.INFO)


def generate_trading_signals(data):
    """
    Genererar köp- och säljsignaler baserat på tekniska indikatorer.
    """
    try:
        data["SMA_50"] = data["close"].rolling(window=50).mean()
        data["SMA_200"] = data["close"].rolling(window=200).mean()

        data["signal"] = "HOLD"
        data.loc[data["SMA_50"] > data["SMA_200"], "signal"] = "BUY"
        data.loc[data["SMA_50"] < data["SMA_200"], "signal"] = "SELL"

        logging.info("✅ Köp- och säljsignaler genererade.")
        return data[["date", "close", "SMA_50", "SMA_200", "signal"]]
    except Exception as e:
        logging.error(f"❌ Fel vid signalgenerering: {str(e)}")
        return None


# Exempelanrop
if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=300),
            "close": np.cumsum(np.random.randn(300) * 2 + 100),
        }
    )

    signals = generate_trading_signals(df)
    print("📢 Genererade signaler:")
    print(signals.tail())
