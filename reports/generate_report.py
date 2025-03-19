import logging

import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

# Konfigurera loggning
logging.basicConfig(filename="generate_report.log", level=logging.INFO)


def create_performance_chart(trade_log, output_file="performance_chart.png"):
    """
    Skapar en diagram över handelsstrategins resultat.
    """
    try:
        trade_log["Cumulative Returns"] = (1 + trade_log["return"]).cumprod()
        plt.figure(figsize=(10, 5))
        plt.plot(
            trade_log["Cumulative Returns"], label="Portföljutveckling", color="blue"
        )
        plt.axhline(y=1, color="gray", linestyle="--", label="Startvärde")
        plt.legend()
        plt.title("Handelsstrategins Prestanda")
        plt.xlabel("Handel")
        plt.ylabel("Avkastning")
        plt.savefig(output_file)
        plt.close()
        logging.info("✅ Prestandadiagram genererat.")
    except Exception as e:
        logging.error(f"❌ Fel vid skapande av prestandadiagram: {str(e)}")


def generate_pdf_report(trade_log, filename="trading_report.pdf"):
    """
    Genererar en PDF-rapport med resultat av AI-tradingstrategin.
    """
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "AI Trading Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Antal Affärer: {len(trade_log)}", ln=True)
        pdf.cell(200, 10, f"Total Avkastning: {trade_log['return'].sum():.2%}", ln=True)
        pdf.cell(200, 10, f"Win Rate: {(trade_log['return'] > 0).mean():.2%}", ln=True)

        pdf.ln(10)
        pdf.cell(200, 10, "Handelsstrategins Utveckling:", ln=True)
        pdf.image("performance_chart.png", x=10, w=180)

        pdf.output(filename)
        logging.info("✅ PDF-rapport genererad.")
    except Exception as e:
        logging.error(f"❌ Fel vid skapande av PDF-rapport: {str(e)}")


# Exempelanrop
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

    create_performance_chart(trade_log)
    generate_pdf_report(trade_log)
    print("📄 PDF-rapport skapad: trading_report.pdf")
