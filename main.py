import logging
from datetime import datetime

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

def main():
    logging.info("🚀 AI Trading Bot startar...")
    try:
        # 1. Hämta & bearbeta data
        # Exempel: hämtar valutadata för USD/SEK
        market_data = fetch_forex_data("USD", "SEK") or {}
        portfolio_data = fetch_portfolio() or {}
        sentiment = analyze_sentiment() or {"sentiment": "neutral"}
        macro_data = fetch_macro_data() or {}
        news_sentiment = fetch_and_analyze_news("stock market") or {"sentiment": "neutral"}

        # Säkerställ att market_data är en dictionary och har 'close'
        # (fetch_forex_data returnerar t.ex. {"close": 10.25} eller None)
        if not isinstance(market_data, dict) or "close" not in market_data:
            logging.error("❌ 'close' saknas i market_data, kan ej fortsätta.")
            return

        # 2. Förbered data för strategi
        # Här är market_data["close"] en enstaka siffra om du hämtar valutakurs => man kan t.ex. göra en fiktiv lista
        # Om du vill ha tidsserie, anpassa fetch_forex_data att returnera en historik istället.
        # Nedan gör vi en "dummy" för demonstration:
        close_series = [market_data["close"]]  # Exempel: en liten lista
        normalized_data = min_max_normalization(close_series)

        # Beräkna volatilitet (ex. daily, men här har vi en kort fiktiv serie)
        volatility = calculate_daily_volatility(close_series) or 0.0

        # 3. AI Beslutsfattande
        strategy = generate_momentum_strategy(normalized_data, sentiment, macro_data) or {}
        optimal_entry = optimal_entry_exit_strategy(strategy) or {}
        refined_strategy = refine_trading_strategy(optimal_entry) or {}

        # 4. Riskhantering
        stop_loss = adaptive_stop_loss(refined_strategy) or {"stop_loss": 0.0}
        var_analysis = calculate_var(close_series) or 0.0
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
        generate_weekly_market_report(market_data)
        generate_macro_event_impact_report(macro_data)

        # 7. Live Trading Signalering (Telegram)
        bot_token = "DIN_TELEGRAM_BOT_TOKEN"  # Lägg till riktig token
        chat_id = "DIN_CHAT_ID"  # Lägg till riktigt chat-ID
        live_signals = generate_trading_signals(market_data) or []
        
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
