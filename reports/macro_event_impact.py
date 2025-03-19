import logging

# Konfigurera loggning
logging.basicConfig(filename="macro_event_impact.log", level=logging.INFO)


def generate_macro_event_impact_report(event_data):
    """
    Analyserar hur en makrohändelse påverkat marknaden.
    """
    try:
        report = (
            f"📢 Makrohändelsepåverkan:\n"
            f"Händelse: {event_data['event']}\n"
            f"S&P 500 påverkan: {event_data['sp500_impact']}%\n"
            f"Obligationsmarknadens respons: {event_data['bond_market']}\n"
            f"USD-rörelse: {event_data['usd_movement']}%"
        )

        logging.info("✅ Makrohändelserapport genererad.")
        return report
    except Exception as e:
        logging.error(f"❌ Fel vid skapande av makrohändelserapport: {str(e)}")
        return None


# Exempelanrop
if __name__ == "__main__":
    event_data = {
        "event": "Fed höjde räntan med 0.25%",
        "sp500_impact": -1.8,
        "bond_market": "Räntorna steg",
        "usd_movement": 0.5,
    }
    report = generate_macro_event_impact_report(event_data)
    print(report)
