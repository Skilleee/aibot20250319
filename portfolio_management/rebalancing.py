import logging

import pandas as pd

# Konfigurera loggning
logging.basicConfig(filename="rebalancing.log", level=logging.INFO)


def rebalancing(portfolio):
    """
    Rebalanserar portföljen för att säkerställa optimal riskfördelning.
    """
    try:
        num_assets = len(portfolio)
        target_allocation = 1 / num_assets  # Enkel rebalanseringsstrategi
        portfolio["new_allocation"] = target_allocation
        logging.info("✅ Portfölj rebalanserad.")
        return portfolio
    except Exception as e:
        logging.error(f"❌ Fel vid rebalansering: {str(e)}")
        return portfolio


# Exempelanrop
if __name__ == "__main__":
    portfolio = pd.DataFrame(
        {
            "symbol": ["AAPL", "TSLA", "GOOGL", "JPM", "XOM"],
            "sector": ["Tech", "Tech", "Tech", "Finance", "Energy"],
            "allocation": [0.30, 0.25, 0.15, 0.20, 0.10],
        }
    )

    rebalanced_portfolio = rebalancing(portfolio)

    print("📊 Rebalanserad portfölj:")
    print(rebalanced_portfolio)
