import logging

# Konfigurera loggning
logging.basicConfig(filename="hedging_strategies.log", level=logging.INFO)


def suggest_hedging_strategies(portfolio_risk_level):
    """
    Rekommenderar hedging-strategier baserat på risknivå.
    """
    try:
        if portfolio_risk_level > 0.7:
            strategy = (
                "Överväg att investera i guld, obligationer eller defensiva aktier."
            )
        elif portfolio_risk_level > 0.4:
            strategy = "Diversifiera med lågrisk-ETF:er och kassareserver."
        else:
            strategy = "Portföljens risk är låg. Ingen ytterligare hedging krävs."

        logging.info("✅ Hedging-strategier föreslagna.")
        return strategy
    except Exception as e:
        logging.error(f"❌ Fel vid hedging-analys: {str(e)}")
        return None


# Exempelanrop
if __name__ == "__main__":
    portfolio_risk = 0.65
    strategy = suggest_hedging_strategies(portfolio_risk)
    print(f"📢 Hedging-strategi: {strategy}")
