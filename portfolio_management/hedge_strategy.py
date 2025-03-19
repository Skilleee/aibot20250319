import logging

# Konfigurera loggning
logging.basicConfig(filename="hedge_strategy.log", level=logging.INFO)


def hedge_strategy(risk_level):
    """
    Föreslår hedge-strategier baserat på risknivån.
    """
    try:
        if risk_level > 0.7:
            hedge = "Öka guld och obligationer"
        elif risk_level < 0.3:
            hedge = "Öka aktieexponering"
        else:
            hedge = "Neutral strategi"

        logging.info(f"✅ Hedge-strategi vald: {hedge}")
        return hedge
    except Exception as e:
        logging.error(f"❌ Fel vid val av hedge-strategi: {str(e)}")
        return "Ingen strategi"


# Exempelanrop
if __name__ == "__main__":
    risk_level = 0.75  # Simulerad risknivå
    hedge = hedge_strategy(risk_level)
    print(f"🛡 Hedge-strategi: {hedge}")
