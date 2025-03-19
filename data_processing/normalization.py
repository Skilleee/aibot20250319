import logging
from datetime import datetime

import numpy as np

# Konfigurera loggning
logging.basicConfig(filename="normalization.log", level=logging.INFO)


# Funktion för att normalisera data med min-max scaling
def min_max_normalization(data):
    """
    Normaliserar data genom att skala värden mellan 0 och 1 med Min-Max Scaling.
    """
    try:
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        logging.info(f"[{datetime.now()}] ✅ Min-Max Normalisering genomförd.")
        return normalized_data
    except Exception as e:
        logging.error(f"[{datetime.now()}] ❌ Fel vid Min-Max normalisering: {str(e)}")
        return None


# Funktion för att standardisera data med Z-score
def z_score_standardization(data):
    """
    Standardiserar data genom att använda Z-score normalisering.
    """
    try:
        standardized_data = (data - np.mean(data)) / np.std(data)
        logging.info(f"[{datetime.now()}] ✅ Z-score Standardisering genomförd.")
        return standardized_data
    except Exception as e:
        logging.error(
            f"[{datetime.now()}] ❌ Fel vid Z-score standardisering: {str(e)}"
        )
        return None


# Funktion för att hantera outliers med winsorization
def winsorize_data(data, limit=0.05):
    """
    Hanterar outliers genom att använda Winsorization, där extrema värden ersätts med percentilgränser.
    """
    try:
        lower_bound = np.percentile(data, limit * 100)
        upper_bound = np.percentile(data, (1 - limit) * 100)
        winsorized_data = np.clip(data, lower_bound, upper_bound)
        logging.info(
            f"[{datetime.now()}] ✅ Winsorization genomförd med gräns {limit}."
        )
        return winsorized_data
    except Exception as e:
        logging.error(f"[{datetime.now()}] ❌ Fel vid Winsorization: {str(e)}")
        return None


# Funktion för att skala data till en logaritmisk skala
def log_scale_data(data):
    """
    Skalar data med en logaritmisk transformation för att hantera skevhet i fördelningen.
    """
    try:
        log_scaled_data = np.log1p(data)
        logging.info(f"[{datetime.now()}] ✅ Logaritmisk skalning genomförd.")
        return log_scaled_data
    except Exception as e:
        logging.error(f"[{datetime.now()}] ❌ Fel vid logaritmisk skalning: {str(e)}")
        return None


# Exempelanrop
if __name__ == "__main__":
    sample_data = np.array([100, 200, 300, 400, 500, 1000, 5000])

    min_max_data = min_max_normalization(sample_data)
    print(f"📏 Min-Max Normaliserad data: {min_max_data}")

    z_score_data = z_score_standardization(sample_data)
    print(f"📊 Z-score Standardiserad data: {z_score_data}")

    winsorized_data = winsorize_data(sample_data)
    print(f"🔍 Winsoriserad data: {winsorized_data}")

    log_scaled_data = log_scale_data(sample_data)
    print(f"📈 Logaritmiskt skalad data: {log_scaled_data}")
