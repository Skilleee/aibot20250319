import logging

# Konfigurera loggning
logging.basicConfig(filename="ai_strategy_evaluation.log", level=logging.INFO)


def evaluate_ai_strategy(strategy_data):
    """
    Utvärderar AI-strategins prestanda och träffsäkerhet.
    """
    try:
        report = (
            f"📊 AI-strategiutvärdering:\n"
            f"Momentum-strategi träffsäkerhet: {strategy_data['momentum_accuracy']}%\n"
            f"Mean-reversion-strategi träffsäkerhet: {strategy_data['mean_reversion_accuracy']}%\n"
            f"Totalt antal affärer: {strategy_data['total_trades']}"
        )

        logging.info("✅ AI-strategiutvärdering genererad.")
        return report
    except Exception as e:
        logging.error(f"❌ Fel vid skapande av strategiutvärdering: {str(e)}")
        return None


# Exempelanrop
if __name__ == "__main__":
    strategy_data = {
        "momentum_accuracy": 78,
        "mean_reversion_accuracy": 65,
        "total_trades": 120,
    }
    report = evaluate_ai_strategy(strategy_data)
    print(report)
