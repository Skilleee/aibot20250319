import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Exempeldata för demonstration ---
# I en riktig implementation hämtar du data från din AI-bot, Google Sheets, eller en databas.

# Simulerad portfölj
portfolio_data = pd.DataFrame({
    "Symbol": ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"],
    "Allocation": [0.20, 0.15, 0.10, 0.25, 0.30],
    "Sector": ["Tech", "Tech", "Tech", "Tech", "Tech"],
})

# Simulerad historisk prisdata (line chart)
dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
prices = np.cumsum(np.random.randn(200) * 2 + 100)  # Slumpad trend

def main():
    st.title("AI-Tradingbot – Interaktiv Dashboard")
    st.write("En enkel demo som visar hur du kan visualisera din portfölj och marknadsdata i realtid.")

    # 1. Visa portföljens aktuella allokering
    st.subheader("Portföljöversikt")
    st.dataframe(portfolio_data)

    # 2. Visa en slider för att filtrera antal dagar att visa
    st.subheader("Historisk prisutveckling")
    max_days = st.slider("Välj antal dagar att visa:", min_value=50, max_value=200, value=100)
    filtered_dates = dates[-max_days:]
    filtered_prices = prices[-max_days:]

    # 3. Rita en enkel linjegraf för den historiska prisutvecklingen
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(filtered_dates, filtered_prices, color="blue", label="Pris")
    ax.set_title("Exempel på prisutveckling")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Pris")
    ax.legend()
    st.pyplot(fig)

    # 4. En sektion för att visa enkla nyckeltal eller AI-rekommendationer
    st.subheader("AI-rekommendationer")
    st.write("Här kan du visa de senaste signalerna som din AI-bot genererat.")
    example_signals = [
        {"Symbol": "AAPL", "Signal": "BUY", "Confidence": 0.85},
        {"Symbol": "TSLA", "Signal": "SELL", "Confidence": 0.60},
        {"Symbol": "NVDA", "Signal": "HOLD", "Confidence": 0.75},
    ]
    st.dataframe(example_signals)

    # 5. Möjlighet att ladda upp/visa annan data i realtid (placeholder)
    st.subheader("Live-data eller andra analyser")
    st.write("Här kan du koppla realtidsdata från WebSockets eller ditt eget API.")

    st.info("Dashboarden är bara ett exempel – anpassa den efter din AI-bots behov!")

if __name__ == "__main__":
    main()
