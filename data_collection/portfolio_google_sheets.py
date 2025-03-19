import logging
from datetime import datetime

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Konfiguration för Google Sheets API
SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
CREDENTIALS_FILE = "credentials.json"  # Ersätt med din service account JSON-fil
SPREADSHEET_NAME = "DinPortfölj"  # Ersätt med namnet på ditt Google Sheet

# Skapa en loggfil
logging.basicConfig(filename="portfolio_google_sheets.log", level=logging.INFO)


def authenticate_google_sheets():
    """
    Autentiserar och ansluter till Google Sheets.
    """
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            CREDENTIALS_FILE, SCOPE
        )
        client = gspread.authorize(creds)
        logging.info(f"[{datetime.now()}] ✅ Lyckad autentisering mot Google Sheets")
        return client
    except Exception as e:
        logging.error(f"[{datetime.now()}] ❌ Fel vid autentisering: {str(e)}")
        return None


def fetch_portfolio():
    """
    Hämtar portföljdata från Google Sheets och returnerar en dictionary med innehavet.
    """
    try:
        client = authenticate_google_sheets()
        if client is None:
            return {}

        sheet = client.open(SPREADSHEET_NAME).sheet1  # Öppnar första bladet
        data = sheet.get_all_records()

        portfolio = {}
        for row in data:
            stock = row.get("Stock")
            quantity = row.get("Quantity")
            if stock and quantity:
                portfolio[stock] = quantity

        logging.info(f"[{datetime.now()}] 📄 Hämtade portföljdata: {portfolio}")
        return portfolio
    except Exception as e:
        logging.error(
            f"[{datetime.now()}] ❌ Fel vid hämtning av portföljdata: {str(e)}"
        )
        return {}


def update_portfolio(stock, quantity):
    """
    Uppdaterar innehavet för en aktie i Google Sheets.
    """
    try:
        client = authenticate_google_sheets()
        if client is None:
            return False

        sheet = client.open(SPREADSHEET_NAME).sheet1
        data = sheet.get_all_records()

        # Hitta raden för aktien och uppdatera kvantiteten
        row_index = None
        for i, row in enumerate(data, start=2):  # Start från rad 2 (index 1)
            if row.get("Stock") == stock:
                row_index = i
                break

        if row_index:
            sheet.update_cell(row_index, 2, quantity)  # Uppdatera kvantitet (kolumn 2)
            logging.info(
                f"[{datetime.now()}] 🔄 Uppdaterade {stock} till {quantity} aktier"
            )
            return True
        else:
            logging.warning(
                f"[{datetime.now()}] ⚠️ {stock} hittades ej i portföljen, lägger till ny rad"
            )
            sheet.append_row([stock, quantity])
            return True
    except Exception as e:
        logging.error(
            f"[{datetime.now()}] ❌ Fel vid uppdatering av portfölj: {str(e)}"
        )
        return False


# Exempelanrop
if __name__ == "__main__":
    portfolio = fetch_portfolio()
    print(f"📊 Aktuell portfölj: {portfolio}")

    success = update_portfolio("AAPL", 15)
    if success:
        print("✅ Portföljen har uppdaterats!")
    else:
        print("❌ Misslyckades att uppdatera portföljen!")
