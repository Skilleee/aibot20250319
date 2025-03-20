import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Behåller alla imports från din gamla main.py
from ai_decision_engine.execution_feedback import refine_trading_strategy
from ai_decision_engine.optimal_entry_exit import optimal_entry_exit_strategy
from ai_decision_engine.strategy_generation import generate_momentum_strategy
from data_collection.macro_data import fetch_macro_data
from data_collection.market_data import fetch_forex_data
from data_collection.news_analysis import fetch_and_analyze_news
from data_collection.portfolio_google_sheets import fetch_portfolio
from data_collection.sentiment_analysis import analyze_sentiment
from data_processing.normalization import (
    min_max_normalization,
    z_score_standardization,
    winsorize_data,
    log_scale_data,
)
from data_processing.volatility_analysis import (
    calculate_daily_volatility,
    calculate_annual_volatility,
    analyze_historical_volatility,
    fetch_vix_index,
)
from live_trading.live_signal_generator import generate_trading_signals
from live_trading.telegram_signal_sender import send_telegram_signal
from portfolio_management.hedge_strategy import hedge_strategy
from portfolio_management.rebalancing import rebalancing
from reports.generate_report import generate_pdf_report
from reports.macro_event_impact import generate_macro_event_impact_report
from reports.weekly_market_report import generate_weekly_market_report
from risk_management.adaptive_stop_loss import adaptive_stop_loss
from risk_management.monte_carlo_simulation import monte_carlo_simulation
from risk_management.value_at_risk import calculate_var
from utils.bot_scheduler import trading_routine, schedule_tasks
from utils.process_manager import manage_processes

# --- Ny import för att ladda RL-modellen ---
from stable_baselines3 import PPO
# Om du vill backtesta RL i main, importera ex. backtest_rl_agent
# from ai_learning.backtest_rl import backtest_rl_agent

# Konfigurera loggning
logging.basicConfig(
    filename="ai_trading_bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Ladda in miljövariabler från .env ===
load_dotenv()

def main():
    logging.info("🚀 AI Trading Bot startar...")

    # 0. Försök ladda en redan tränad RL-modell (om existerar)
    rl_model = None
    try:
        rl_model = PPO.load("rl_trading_model.zip")
        logging.info("✅ RL-modell (PPO) laddad från rl_trading_model.zip")
    except Exception as e:
        logging.warning(f"⚠️ Ingen RL-modell hittad eller kunde ej laddas: {str(e)}")

    try:
        # 1. Hämta & bearbeta data
        #    Hämtar en tidsserie (ex. en månad) för USD/SEK
        market_data = fetch_forex_data("USD", "SEK", period="1mo") or {}
        portfolio_data = fetch_portfolio() or {}
        sentiment = analyze_sentiment() or {"sentiment": "neutral"}
        macro_data = fetch_macro_data() or {}
        news_sentiment = fetch_and_analyze_news("stock market") or {"sentiment": "neutral"}

        # I nya fetch_forex_data() returneras {"history": <DataFrame>}
        df = market_data.get("history")
        if df is None or df.empty:
            logging.error("❌ Ingen valutahistorik för USD/SEK, kan ej fortsätta.")
            return

        # 2. Förbered data för strategi
        close_series = df["Close"]  # Pandas Series med dagliga stängningskurser
        normalized_data = min_max_normalization(close_series.values)
        volatility = calculate_daily_volatility(close_series.values) or 0.0

        # 3. AI-beslutsfattande (klassiskt momentum + optional RL)
        # -------------------------------------------------------
        # a) Generera klassiska strategier (momentum, etc.)
        strategy_input = {"close": normalized_data}
        strategy = generate_momentum_strategy(strategy_input, sentiment, macro_data) or {}
        optimal_entry = optimal_entry_exit_strategy(strategy) or {}
        refined_strategy = refine_trading_strategy(optimal_entry) or {}

        # b) Om vi har RL-modell => generera RL-action
        #    Ex: Skapa en observation (pris, momentum, etc.) i en array
        #    Nedan är en enkel exempellösning
        rl_action = None
        if rl_model is not None:
            # Ex. observation [close, momentum, volume, balance, holding]
            # Här har vi inte balance/holding i main, men vi fejk-lägger in 10000.0 respektive 0.0
            last_row = df.iloc[-1]
            obs = np.array([
                last_row["Close"],
                last_row.get("momentum", 0.0),
                last_row.get("volume", 500.0),
                10000.0,   # Ex. balance
                0.0        # holding
            ], dtype=np.float32)

            action, _ = rl_model.predict(obs, deterministic=True)
            # 0=HOLD, 1=BUY, 2=SELL (beroende på hur du definierat i TradingEnv)
            rl_action = action
            logging.info(f"RL-agent föreslår action: {rl_action}")

        # 4. Riskhantering
        stop_loss = adaptive_stop_loss(refined_strategy) or {"stop_loss": 0.0}
        var_analysis = calculate_var(close_series.values) or 0.0
        monte_carlo_sim = monte_carlo_simulation(100000, 0.07, 0.2) or {}

        # 5. Portföljhantering
        if not isinstance(portfolio_data, dict):
            logging.error("❌ Fel: portfolio_data är ogiltig. Kan inte fortsätta med portföljhantering.")
            return

        rebalanced_portfolio = rebalancing(portfolio_data, refined_strategy) or {}
        hedging_plan = hedge_strategy(rebalanced_portfolio)
        if isinstance(hedging_plan, str):
            hedging_plan = {"hedging_plan": hedging_plan}

        # 6. Generera rapporter
        generate_pdf_report(rebalanced_portfolio)
        generate_weekly_market_report(macro_data)
        generate_macro_event_impact_report(macro_data)

        # 7. Live Trading Signalering (Telegram)
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

        live_signals = generate_trading_signals(macro_data) or []

        # Om RL action finns => logga och inkludera den i "live_signals"
        if rl_action is not None:
            rl_msg = f"RL-agent action: {rl_action}"
            live_signals.append(rl_msg)

        if bot_token and chat_id:
            send_telegram_signal(bot_token, chat_id, "📢 Live Trading Signal", live_signals)
        else:
            logging.warning("⚠️ Telegram-token eller chat-ID saknas, kan ej skicka signaler.")

        # 8. Systemhantering
        schedule_tasks()
        manage_processes()

        logging.info("✅ AI Trading Bot körs och analyserar marknaden.")

    except Exception as e:
        logging.error(f"❌ Fel i main(): {str(e)}")


if __name__ == "__main__":
    main()
