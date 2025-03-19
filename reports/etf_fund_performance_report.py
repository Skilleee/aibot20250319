import logging

# Konfigurera loggning
logging.basicConfig(filename="etf_fund_performance_report.log", level=logging.INFO)


def generate_etf_fund_performance_report(fund_data):
    """
    Skapar en rapport över avkastning och risknivåer i fonder och ETF:er.
    """
    try:
        best_fund = max(fund_data, key=lambda x: fund_data[x]["return"])

        report = (
            f"📊 ETF & Fondrapport:\n"
            f"Bästa fonden: {best_fund} ({fund_data[best_fund]['return']}% avkastning)\n"
            f"Sharpe-kvot: {fund_data[best_fund]['sharpe_ratio']}"
        )

        logging.info("✅ ETF- och fondrapport genererad.")
        return report
    except Exception as e:
        logging.error(f"❌ Fel vid skapande av fondrapport: {str(e)}")
        return None


# Exempelanrop
if __name__ == "__main__":
    fund_data = {
        "SPY": {"return": 6.3, "sharpe_ratio": 1.2},
        "VOO": {"return": 5.8, "sharpe_ratio": 1.1},
        "ARKK": {"return": 3.2, "sharpe_ratio": 0.8},
    }
    report = generate_etf_fund_performance_report(fund_data)
    print(report)
