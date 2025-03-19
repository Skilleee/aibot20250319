import logging
from datetime import datetime

import requests

# Konfigurera loggning
logging.basicConfig(filename="macro_data.log", level=logging.INFO)

# API-konfiguration (valfri expansion till exempelvis FRED, ECB eller IMF)
MACRO_API_URL = "https://www.alphavantage.co/query"
API_KEY = "YOUR_API_KEY"  # Ersätt med din API-nyckel


# Funktion för att hämta makroekonomiska indikatorer
def fetch_macro_data(indicator):
    """
    Hämtar makroekonomiska data såsom BNP, räntor och inflation.
    """
    try:
        params = {"function": indicator, "apikey": API_KEY}
        response = requests.get(MACRO_API_URL, params=params)
        data = response.json()

        if "data" in data:
            macro_value = data["data"]
            logging.info(f"[{datetime.now()}] ✅ Hämtade {indicator}: {macro_value}")
            return macro_value
        else:
            logging.warning(f"[{datetime.now()}] ⚠️ Ingen data hittades för {indicator}")
            return None
    except Exception as e:
        logging.error(
            f"[{datetime.now()}] ❌ Fel vid hämtning av {indicator}: {str(e)}"
        )
        return None


# Funktion för att hämta räntor
def fetch_interest_rates():
    """
    Hämtar aktuella centralbanksräntor.
    """
    return fetch_macro_data("FEDERAL_FUNDS_RATE")


# Funktion för att hämta inflation
def fetch_inflation():
    """
    Hämtar aktuella inflationsdata.
    """
    return fetch_macro_data("INFLATION")


# Funktion för att hämta BNP-tillväxt
def fetch_gdp_growth():
    """
    Hämtar BNP-tillväxt.
    """
    return fetch_macro_data("REAL_GDP")


# Funktion för att hämta arbetslöshetsstatistik
def fetch_unemployment_rate():
    """
    Hämtar arbetslöshetsnivå.
    """
    return fetch_macro_data("UNEMPLOYMENT_RATE")


# Exempelanrop
if __name__ == "__main__":
    interest_rate = fetch_interest_rates()
    print(f"🏦 Aktuell ränta: {interest_rate}")

    inflation = fetch_inflation()
    print(f"📈 Inflation: {inflation}")

    gdp_growth = fetch_gdp_growth()
    print(f"📊 BNP-tillväxt: {gdp_growth}")

    unemployment = fetch_unemployment_rate()
    print(f"👥 Arbetslöshet: {unemployment}")
