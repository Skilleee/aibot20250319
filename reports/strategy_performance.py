import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Konfigurera loggning
logging.basicConfig(filename="strategy_performance.log", level=logging.INFO)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Beräknar Sharpe-kvoten för att mäta riskjusterad avkastning.
    """
    try:
        # risk_free_rate antas vara årlig => dela med 252 för daglig
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        logging.info(f"✅ Sharpe Ratio beräknad: {sharpe_ratio:.2f}")
        return sharpe_ratio
    except Exception as e:
        logging.error(f"❌ Fel vid beräkning av Sharpe Ratio: {str(e)}")
        return None

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """
    Beräknar Sortino-kvoten, som fokuserar på nedsiderisk.
    """
    try:
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns)
        excess_returns = returns - risk_free_rate / 252
        sortino_ratio = np.mean(excess_returns) / downside_deviation
        logging.info(f"✅ Sortino Ratio beräknad: {sortino_ratio:.2f}")
        return sortino_ratio
    except Exception as e:
        logging.error(f"❌ Fel vid beräkning av Sortino Ratio: {str(e)}")
        return None

def plot_strategy_performance(trade_log, output_file="strategy_performance.png"):
    """
    Skapar ett diagram över handelsstrategins avkastning över tiden.
    """
    try:
        trade_log["Cumulative Returns"] = (1 + trade_log["return"]).cumprod()
        plt.figure(figsize=(10, 5))
        plt.plot(
            trade_log["Cumulative Returns"], label="Strategins Avkastning", color="blue"
        )
        plt.axhline(y=1, color="gray", linestyle="--", label="Startvärde")
        plt.legend()
        plt.title("Strategins Prestanda")
        plt.xlabel("Handel")
        plt.ylabel("Avkastning")
        plt.savefig(output_file)
        plt.close()
        logging.info("✅ Strategins prestandadiagram genererat.")
    except Exception as e:
        logging.error(f"❌ Fel vid skapande av prestandadiagram: {str(e)}")

# ------------------- NYA FUNKTIONER -------------------

def log_ai_recommendation(recommendations, trade_log_file="trade_log.csv"):
    """
    Lagrar AI:s rekommendationer i en CSV-fil (trade_log).
    Varje rad kan exempelvis innehålla:
      - symbol: Aktiens ticker
      - signal: 'BUY', 'SELL', 'HOLD' etc.
      - recommended_price: Priset när AI:n gav rekommendationen
      - timestamp: När rekommendationen gavs
    """
    try:
        df = pd.DataFrame(recommendations)
        df["timestamp"] = datetime.now()
        # Om filen redan finns, append till den, annars skapa ny fil
        header = not pd.io.common.file_exists(trade_log_file)
        df.to_csv(trade_log_file, mode="a", header=header, index=False)
        logging.info(f"✅ {len(df)} AI-rekommendation(er) loggad(e) i {trade_log_file}.")
    except Exception as e:
        logging.error(f"❌ Fel vid loggning av AI-rekommendationer: {str(e)}")

def simulate_pl_from_log(trade_log):
    """
    En enkel simulering av P/L baserat på in- och utpriser i trade_log.
    trade_log förväntas innehålla kolumnerna:
      - entry_price
      - exit_price
      - (ev. tidsdata)
    Returnerar total avkastning och en DataFrame med per-trade P/L.
    """
    try:
        # Anta att 'return' redan finns eller kan beräknas:
        if "return" not in trade_log.columns:
            trade_log["return"] = (trade_log["exit_price"] - trade_log["entry_price"]) / trade_log["entry_price"]

        total_return = (1 + trade_log["return"]).prod() - 1
        logging.info(f"✅ Simulerad totalavkastning: {total_return:.2%}")
        return total_return, trade_log
    except Exception as e:
        logging.error(f"❌ Fel vid P/L-simulering: {str(e)}")
        return None, trade_log

# ------------------- EXEMPELANROP --------------------
if __name__ == "__main__":
    # Simulerad handelslogg
    trade_log = pd.DataFrame(
        {
            "symbol": ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"],
            "entry_price": [150, 700, 250, 300, 2800],
            "exit_price": [155, 680, 270, 310, 2900],
            "return": [0.033, -0.028, 0.08, 0.033, 0.035],
            "trade_date": pd.date_range(start="2023-01-01", periods=5),
        }
    )

    # 1. Beräkna Sharpe & Sortino
    sharpe = calculate_sharpe_ratio(trade_log["return"])
    sortino = calculate_sortino_ratio(trade_log["return"])
    print(f"📊 Sharpe Ratio: {sharpe:.2f}")
    print(f"📉 Sortino Ratio: {sortino:.2f}")

    # 2. Generera diagram
    plot_strategy_performance(trade_log)

    # 3. Exempel: logga nya AI-rekommendationer
    sample_recs = [
        {"symbol": "TSLA", "signal": "BUY", "recommended_price": 210.5},
        {"symbol": "AAPL", "signal": "SELL", "recommended_price": 155.2},
    ]
    log_ai_recommendation(sample_recs, "trade_log.csv")

    # 4. Simulera P/L från trade_log
    total_return, updated_log = simulate_pl_from_log(trade_log)
    print(f"📈 Totalavkastning: {total_return:.2%}")
