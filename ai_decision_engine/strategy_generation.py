import logging
from datetime import datetime

import numpy as np
import pandas as pd

# För Prophet:
try:
    from prophet import Prophet
except ImportError:
    from fbprophet import Prophet

# Konfigurera loggning
logging.basicConfig(filename="strategy_generation.log", level=logging.INFO)


# Funktion för att generera en enkel momentum-baserad strategi
def generate_momentum_strategy(data, sentiment=None, macro_data=None):
    """
    Genererar en momentum-baserad strategi baserat på sentiment och prisdata.
    """
    if "close" in data and (isinstance(data["close"], pd.Series) or isinstance(data["close"], np.ndarray)):
        # Exempelstrategi: om positivt sentiment > negativt => 'buy', annars 'sell'
        if sentiment is not None:
            strategy = {"signal": "buy" if sentiment.get("positive", 0) > sentiment.get("negative", 0) else "sell"}
        else:
            # Om inget sentiment finns, slumpa signal som exempel.
            strategy = {"signal": np.random.choice(["buy", "sell"]) }
        return pd.DataFrame({"close": data["close"], "signal": strategy["signal"]}, index=data.index)
    else:
        return None


# Funktion för att generera en mean reversion-strategi
def generate_mean_reversion_strategy(data, window=50, threshold=2):
    """
    Genererar en mean reversion-strategi genom att analysera Bollinger Bands.
    """
    try:
        df = data.copy()
        df["moving_avg"] = df["close"].rolling(window=window).mean()
        df["std_dev"] = df["close"].rolling(window=window).std()
        df["upper_band"] = df["moving_avg"] + (threshold * df["std_dev"])
        df["lower_band"] = df["moving_avg"] - (threshold * df["std_dev"])
        df["signal"] = np.where(
            df["close"] < df["lower_band"],
            1,  # Köp
            np.where(df["close"] > df["upper_band"], -1, 0),  # Sälj, annars neutral
        )

        logging.info(
            f"[{datetime.now()}] ✅ Mean reversion-strategi genererad med {window}-dagars Bollinger Bands."
        )
        return df[["close", "moving_avg", "upper_band", "lower_band", "signal"]]
    except Exception as e:
        logging.error(
            f"[{datetime.now()}] ❌ Fel vid generering av mean reversion-strategi: {str(e)}"
        )
        return None


# Funktion för att kombinera strategier
def combine_strategies(momentum_data, mean_reversion_data):
    """
    Kombinerar momentum- och mean reversion-strategier för att skapa en hybridstrategi.
    """
    try:
        # Säkerställ att båda dataframes har samma index
        combined_data = momentum_data.copy()
        if "signal" not in combined_data.columns:
            combined_data["signal"] = 0

        # Se till att vi kan addera signaler
        if mean_reversion_data is not None and "signal" in mean_reversion_data.columns:
            combined_data["combined_signal"] = combined_data["signal"] + mean_reversion_data["signal"].fillna(0)
        else:
            combined_data["combined_signal"] = combined_data["signal"]

        logging.info(
            f"[{datetime.now()}] ✅ Strategier kombinerade till en hybridmodell."
        )
        return combined_data[["close", "combined_signal"]]
    except Exception as e:
        logging.error(
            f"[{datetime.now()}] ❌ Fel vid kombination av strategier: {str(e)}"
        )
        return None


def generate_future_price_forecast(data, forecast_days=365):
    """
    Tar in en DataFrame med kolumnerna ["date", "close"] och genererar en framtidsprognos
    över `forecast_days` dagar med hjälp av Prophet. Returnerar en DataFrame med kolumnen "yhat".
    """
    try:
        df = data.copy()
        # Prophet kräver kolumnerna ds (datum) och y (stängningskurs)
        df = df.rename(columns={"date": "ds", "close": "y"})

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        logging.info(
            f"[{datetime.now()}] ✅ Lyckades generera tidsserieprognos för {forecast_days} dagar."
        )
        return forecast  # Innehåller kolumner som [ds, yhat, yhat_lower, yhat_upper]
    except Exception as e:
        logging.error(
            f"[{datetime.now()}] ❌ Fel vid generering av framtidsprognos: {str(e)}"
        )
        return pd.DataFrame()


def extract_price_targets(forecast):
    """
    Tar in en Prophet-prognos (DataFrame) och returnerar riktpriser för ~3m (90d), ~6m (180d)
    och ~12m (365d) från "idag".
    """
    if forecast.empty:
        return {"3m": None, "6m": None, "12m": None}

    # Filtrera framtida datum
    future_forecast = forecast[forecast["ds"] >= datetime.now()]

    # Skapa placeholders
    price_targets = {"3m": None, "6m": None, "12m": None}

    # 3m: ca 90 dagar in i framtiden
    forecast_3m = future_forecast.head(90)
    if len(forecast_3m) == 90:
        price_targets["3m"] = float(forecast_3m["yhat"].iloc[-1])

    # 6m: ca 180 dagar
    forecast_6m = future_forecast.head(180)
    if len(forecast_6m) == 180:
        price_targets["6m"] = float(forecast_6m["yhat"].iloc[-1])

    # 12m: ca 365 dagar
    forecast_12m = future_forecast.head(365)
    if len(forecast_12m) == 365:
        price_targets["12m"] = float(forecast_12m["yhat"].iloc[-1])

    return price_targets


# Exempelanrop
if __name__ == "__main__":
    # Simulerad prisdata
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    prices = np.cumsum(np.random.randn(200) * 2 + 100)
    stock_data = pd.DataFrame({"date": dates, "close": prices})
    stock_data.set_index("date", inplace=True)

    # Momentum-strategi
    sentiment_data = {"positive": 0.6, "negative": 0.4}
    momentum_strategy = generate_momentum_strategy(stock_data, sentiment_data)
    print(f"\n📈 Momentum-strategi (huvud):")
    print(momentum_strategy.tail())

    # Mean reversion-strategi
    mean_reversion_strategy = generate_mean_reversion_strategy(stock_data)
    print(f"\n📊 Mean Reversion-strategi:")
    print(mean_reversion_strategy.tail())

    # Hybridstrategi
    combined_strategy = combine_strategies(momentum_strategy, mean_reversion_strategy)
    print(f"\n🔀 Hybridstrategi:")
    print(combined_strategy.tail())

    # Generera framtidsprognos med Prophet
    df_for_forecast = pd.DataFrame({"date": dates, "close": prices})  # Prophet behöver date som kolumn
    forecast = generate_future_price_forecast(df_for_forecast, forecast_days=365)
    if not forecast.empty:
        price_targets = extract_price_targets(forecast)
        print(f"\n💡 Riktpriser (3m, 6m, 12m): {price_targets}")
    else:
        print("\n⚠️ Kunde inte generera prognos")