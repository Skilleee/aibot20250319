import logging

# Konfigurera loggning
logging.basicConfig(filename="monthly_performance_report.log", level=logging.INFO)


def generate_monthly_performance_report(portfolio_data):
    """
    Genererar en månadsrapport över portföljens prestanda.
    """
    try:
        report = (
            f"📈 Månadsrapport:\n"
            f"Portföljavkastning: {portfolio_data['portfolio_return']}%\n"
            f"S&P 500 jämförelse: {portfolio_data['sp500']}%\n"
            f"Bästa aktie: {portfolio_data['best_performer']} ({portfolio_data['best_return']}%)\n"
            f"Sämsta aktie: {portfolio_data['worst_performer']} ({portfolio_data['worst_return']}%)"
        )

        logging.info("✅ Månadsrapport genererad.")
        return report
    except Exception as e:
        logging.error(f"❌ Fel vid skapande av månadsrapport: {str(e)}")
        return None


# Exempelanrop
if __name__ == "__main__":
    portfolio_data = {
        "portfolio_return": 5.2,
        "sp500": 4.3,
        "best_performer": "AAPL",
        "best_return": 8.1,
        "worst_performer": "TSLA",
        "worst_return": -2.3,
    }
    report = generate_monthly_performance_report(portfolio_data)
    print(report)
