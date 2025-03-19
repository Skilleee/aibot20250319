import os
from dotenv import load_dotenv
import logging
from datetime import datetime

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

# Konfigurera loggning
logging.basicConfig(
    filename="ai_trading_bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === 1. Ladda in miljövariabler från .env ===
load_dotenv()

def main():
    logging.info("🚀 AI Trading Bot startar...")
    try:
        # 1. Hämta & bearbeta data
        # Uppdaterat: Hämtar en tidsserie (ex. en månad) för USD/SEK
        market_data = fetch_forex_data("USD", "SEK", period="1mo") or {}
        portfolio_data = fetch_portfolio() or {}
        sentiment = analyze_sentiment() or {"sentiment": "neutral"}
        macro_data = fetch_macro_data() or {}
        news_sentiment = fetch_and_analyze_news("stock market") or {"sentiment": "neutral"}

        # I nya fetch_forex_data() returneras {"history": <DataFrame>}
        df = market_data.get("history")
        if df is None or df.empty:
            logging.error("❌ Ingen valutahistorik tillgänglig för USD/SEK, kan ej fortsätta.")
            return

        # 2. Förbered data för strategi
        # Använd stängningskurser som tidsserie
        close_series = df["Close"]  # Pandas Series med dagliga stängningskurser

        # Normalisera (om 'min_max_normalization' förväntar sig en array/list)
        normalized_data = min_max_normalization(close_series.values)

        # Exempel på volatilitet med daily (kunde även anropa annual, men daily är i koden)
        volatility = calculate_daily_volatility(close_series.values) or 0.0

        # 3. AI-beslutsfattande
        # generate_momentum_strategy() tar ofta en dict {"close": ...}, vi skickar in normaliserade data
        strategy_input = {"close": normalized_data}
        strategy = generate_momentum_strategy(strategy_input, sentiment, macro_data) or {}
        optimal_entry = optimal_entry_exit_strategy(strategy) or {}
        refined_strategy = refine_trading_strategy(optimal_entry) or {}

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
        generate_weekly_market_report(macro_data)  # ev. market_data om rapporten behöver rådata
        generate_macro_event_impact_report(macro_data)

        # 7. Live Trading Signalering (Telegram)
        # === Hämta tokens från .env via os.getenv ===
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        # Om du behöver Twitter:
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

        live_signals = generate_trading_signals(macro_data) or []

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
