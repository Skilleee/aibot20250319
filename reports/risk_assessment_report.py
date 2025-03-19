import logging

# Konfigurera loggning
logging.basicConfig(filename="risk_assessment_report.log", level=logging.INFO)


def generate_risk_assessment_report(risk_data):
    """
    Skapar en rapport om portföljens risknivå och varningar.
    """
    try:
        report = (
            f"⚠️ Riskanalys:\n"
            f"Portföljens volatilitet: {risk_data['volatility']}%\n"
            f"Max drawdown: {risk_data['max_drawdown']}%\n"
            f"Risknivå: {risk_data['risk_level']}"
        )

        logging.info("✅ Riskanalysrapport genererad.")
        return report
    except Exception as e:
        logging.error(f"❌ Fel vid skapande av riskrapport: {str(e)}")
        return None


# Exempelanrop
if __name__ == "__main__":
    risk_data = {"volatility": 25, "max_drawdown": -15, "risk_level": "Hög"}
    report = generate_risk_assessment_report(risk_data)
    print(report)
