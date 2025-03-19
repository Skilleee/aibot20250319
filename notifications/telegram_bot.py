import logging

import requests

# Konfigurera loggning
logging.basicConfig(filename="telegram_notifications.log", level=logging.INFO)


def send_telegram_notification(message, bot_token, chat_id):
    """
    Skickar en notis till Telegram-boten.
    """
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        response = requests.post(url, data=data)
        logging.info("✅ Notis skickad till Telegram.")
        return response.json()
    except Exception as e:
        logging.error(f"❌ Fel vid skickning av Telegram-notis: {str(e)}")
        return None


def send_daily_market_report(bot_token, chat_id, market_data):
    """
    Skickar en daglig marknadsrapport till Telegram.
    """
    try:
        summary = (
            f"📊 Daglig marknadsrapport:\n"
            f"S&P 500: {market_data['sp500']}%\n"
            f"Nasdaq: {market_data['nasdaq']}%\n"
            f"Tech-sektorn: {market_data['tech_sector']}%\n"
            f"Sentiment: {market_data['sentiment']}"
        )
        send_telegram_notification(summary, bot_token, chat_id)
    except Exception as e:
        logging.error(f"❌ Fel vid skapande av marknadsrapport: {str(e)}")


def send_risk_alert(bot_token, chat_id, risk_level):
    """
    Skickar en riskvarning om marknaden blir volatil.
    """
    try:
        if risk_level > 0.05:
            message = f"⚠️ Hög volatilitet upptäckt! Risknivå: {risk_level:.2%}. Överväg att minska exponering."
            send_telegram_notification(message, bot_token, chat_id)
    except Exception as e:
        logging.error(f"❌ Fel vid riskvarning: {str(e)}")


def send_portfolio_update(bot_token, chat_id, portfolio_data):
    """
    Skickar en uppdatering om portföljens utveckling.
    """
    try:
        message = "📢 Portföljuppdatering:\n"
        for stock, change in portfolio_data.items():
            message += f"{stock}: {change:.2%}\n"
        send_telegram_notification(message, bot_token, chat_id)
    except Exception as e:
        logging.error(f"❌ Fel vid portföljnotis: {str(e)}")


def send_macro_event_alert(bot_token, chat_id, event):
    """
    Skickar en notifiering vid viktiga makrohändelser.
    """
    try:
        message = f"📢 Makrohändelse: {event}"
        send_telegram_notification(message, bot_token, chat_id)
    except Exception as e:
        logging.error(f"❌ Fel vid makronotis: {str(e)}")


# Exempelanrop
if __name__ == "__main__":
    bot_token = "DIN_TELEGRAM_BOT_TOKEN"
    chat_id = "DIN_CHAT_ID"

    # Exempeldata
    market_data = {
        "sp500": 1.2,
        "nasdaq": 0.8,
        "tech_sector": 1.5,
        "sentiment": "Positiv",
    }
    risk_level = 0.06
    portfolio_data = {"Tesla": -0.05, "Apple": 0.02, "Amazon": 0.03}
    macro_event = "Fed höjde räntan med 0.25%."

    send_daily_market_report(bot_token, chat_id, market_data)
    send_risk_alert(bot_token, chat_id, risk_level)
    send_portfolio_update(bot_token, chat_id, portfolio_data)
    send_macro_event_alert(bot_token, chat_id, macro_event)
