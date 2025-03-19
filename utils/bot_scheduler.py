import logging
import time
import schedule
from ai_decision_engine.strategy_generation import generate_momentum_strategy
from data_collection.market_data import fetch_forex_data
from notifications.telegram_bot import send_telegram_notification

# Konfigurera loggning
logging.basicConfig(filename="bot_scheduler.log", level=logging.INFO)

def trading_routine():
    """
    Huvudrutinen för AI-trading boten. Hämtar marknadsdata, genererar signaler och skickar aviseringar.
    """
    try:
        logging.info("⏳ Startar trading-rutin...")
        market_data = fetch_forex_data()

        # Generera handelsstrategi
        trading_signal = generate_momentum_strategy(market_data)

        # Skicka notis till Telegram
        bot_token = "DIN_TELEGRAM_BOT_TOKEN"  # Lägg till din token
        chat_id = "DIN_CHAT_ID"  # Lägg till ditt chat-ID
        send_telegram_notification(
            f"📢 AI-Trading Uppdatering: {trading_signal}", bot_token, chat_id
        )
        logging.info("✅ Trading-rutin genomförd framgångsrikt.")
    except Exception as e:
        logging.error(f"❌ Fel i trading-rutin: {str(e)}")

def schedule_tasks():
    """
    Schemaläggning av botens uppgifter.
    """
    schedule.every().day.at("09:00").do(trading_routine)  # Körs varje dag kl 09:00
    schedule.every(30).minutes.do(trading_routine)  # Uppdaterar var 30:e minut

    logging.info("🚀 Startar AI-Trading Bot Scheduler...")
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    schedule_tasks()