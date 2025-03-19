import logging
from datetime import datetime
import yfinance as yf

# Skapa en loggfil
logging.basicConfig(filename="market_data.log", level=logging.INFO)

# Funktion för att hämta aktiepriser från Yahoo Finance
def fetch_stock_price(symbol):
    """
    Hämtar den senaste aktiekursen för ett givet aktiesymbol (t.ex. 'AAPL', 'TSLA') från Yahoo Finance.
    """
    try:
        stock = yf.Ticker(symbol)
        latest_price = stock.history(period="1d")["Close"][-1]
        logging.info(
            f"[{datetime.now()}] ✅ Hämtade data för {symbol}: {latest_price} USD"
        )
        return latest_price
    except Exception as e:
        logging.error(f"[{datetime.now()}] ❌ Fel vid hämtning av {symbol}: {str(e)}")
        return None

# Funktion för att hämta flera aktier samtidigt
def fetch_multiple_stocks(symbols):
    """
    Hämtar senaste priser för en lista av aktier från Yahoo Finance.
    """
    stock_prices = {}
    for symbol in symbols:
        price = fetch_stock_price(symbol)
        if price:
            stock_prices[symbol] = price
    return stock_prices

# Funktion för att hämta valutakurser från Yahoo Finance
def fetch_forex_data(base_currency, quote_currency):
    """
    Hämtar realtids växelkurs mellan två valutor, t.ex. USD/SEK från Yahoo Finance.
    """
    try:
        pair = f"{base_currency}{quote_currency}=X"
        stock = yf.Ticker(pair)
        history = stock.history(period="1d")
        if not history.empty:
            latest_price = history["Close"].iloc[-1]
            logging.info(
                f"[{datetime.now()}] 💱 Växelkurs {base_currency}/{quote_currency}: {latest_price}"
            )
            return {"close": latest_price}
        else:
            logging.info(
                f"[{datetime.now()}] 💱 Växelkurs {base_currency}/{quote_currency}: None"
            )
            return {"close": None}
    except Exception as e:
        logging.error(
            f"[{datetime.now()}] ❌ Fel vid hämtning av växelkurs {base_currency}/{quote_currency}: {str(e)}"
        )
        return None

# Funktion för att hämta råvarupriser från Yahoo Finance
def fetch_commodity_price(commodity):
    """
    Hämtar realtidspris för råvaror, t.ex. guld (XAU/USD) eller olja (WTI).
    """
    commodity_map = {"gold": "GC=F", "silver": "SI=F", "oil": "CL=F"}

    if commodity not in commodity_map:
        logging.error(f"[{datetime.now()}] ❌ Ogiltig råvara: {commodity}")
        return None

    return fetch_stock_price(commodity_map[commodity])

# Funktion för att hämta orderflöden (simulerad för Yahoo Finance)
def fetch_order_flow(symbol):
    """
    Simulerar analys av orderflöden och likviditet för en aktie.
    """
    try:
        # Simulerad data för orderflöde
        order_flow_data = {
            "buy_orders": 1200,
            "sell_orders": 800,
            "net_flow": 400,  # Positivt värde indikerar köparövertag
        }
        logging.info(
            f"[{datetime.now()}] 📊 Orderflöde för {symbol}: {order_flow_data}"
        )
        return order_flow_data
    except Exception as e:
        logging.error(
            f"[{datetime.now()}] ❌ Fel vid hämtning av orderflöde för {symbol}: {str(e)}"
        )
        return None

# Funktion för att hämta undervärderade aktier
def fetch_undervalued_stocks(tickers):
    """
    Analyserar en lista av aktier och filtrerar ut de med lågt P/E och hög vinstmarginal.
    """
    undervalued_stocks = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            pe_ratio = info.get("trailingPE", None)
            profit_margin = info.get("profitMargins", None)

            if pe_ratio and profit_margin:
                # Exempel-kriterier: P/E under 15 och vinstmarginal över 10%
                if pe_ratio < 15 and profit_margin > 0.10:
                    undervalued_stocks.append({
                        "symbol": ticker,
                        "pe_ratio": pe_ratio,
                        "profit_margin": profit_margin
                    })
        except Exception as e:
            logging.error(f"[{datetime.now()}] ❌ Fel vid analys av {ticker}: {str(e)}")
    return undervalued_stocks

# Funktion för att skanna flera marknader och hitta guldkorn
def scan_market():
    """
    Skannar utvalda marknader (OMX30, DAX, S&P 500) och identifierar undervärderade aktier.
    """
    markets = {
        "OMX30": ["ERIC-B.ST", "VOLV-B.ST", "NDA-SE.ST", "HM-B.ST"],
        "DAX": ["SAP.DE", "BMW.DE", "DTE.DE", "BAS.DE"],
        "S&P 500": ["AAPL", "TSLA", "NVDA", "MSFT"]
    }

    undervalued = {}
    for market_name, tickers in markets.items():
        found_undervalued = fetch_undervalued_stocks(tickers)
        undervalued[market_name] = found_undervalued
        logging.info(f"[{datetime.now()}] 📊 {market_name} undervärderade aktier: {found_undervalued}")

    return undervalued

# Ny funktion för att hämta de mest omsatta aktierna
def fetch_most_traded_stocks(index_ticker):
    """
    Hämta de 10 mest omsatta aktierna (högst volym) för en given indexsymbol.
    T.ex. '^OMX' för OMX30, '^GDAXI' för DAX, '^GSPC' för S&P 500.
    """
    try:
        index = yf.Ticker(index_ticker)
        tickers = index.options  # Lista över indexets aktier
        volume_data = {}

        # Begränsa för att undvika stora dataanrop
        for ticker in tickers[:20]:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                volume_data[ticker] = data["Volume"].iloc[-1]  # Senaste handelsvolymen

        # Sortera efter volym, högst först
        sorted_stocks = sorted(volume_data.items(), key=lambda x: x[1], reverse=True)
        most_traded = sorted_stocks[:10]
        logging.info(
            f"[{datetime.now()}] 🔥 Mest omsatta aktier för {index_ticker}: {most_traded}"
        )
        return most_traded
    except Exception as e:
        logging.error(
            f"[{datetime.now()}] ❌ Fel vid hämtning av mest omsatta aktier för {index_ticker}: {str(e)}"
        )
        return []

if __name__ == "__main__":
    # Exempelanrop för aktier
    aktier = ["AAPL", "TSLA", "NVDA"]
    priser = fetch_multiple_stocks(aktier)
    print(f"📈 Senaste aktiepriser: {priser}")

    # Exempelanrop för undervärderade aktier
    undervalued_stocks = scan_market()
    print(f"📊 Undervärderade aktier: {undervalued_stocks}")

    # Exempelanrop för de mest omsatta aktierna på OMX (kan vara '^OMX' eller '^OMX30'
    # OBS: Ibland saknas data beroende på hur Yahoo Finance definierar index.options
    # Du kan testa '^GSPC' (S&P 500) eller '^GDAXI' (DAX) också.

    omx_most_traded = fetch_most_traded_stocks('^OMX')
    print(f"🔥 Mest omsatta på OMX: {omx_most_traded}")
