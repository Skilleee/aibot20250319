import numpy as np
import logging

# Konfigurera loggning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def monte_carlo_simulation_normal(initial_value, mean_return, volatility, days=252, simulations=1000):
    """
    Monte Carlo-simulering med normalfördelade dagliga avkastningar.
    
    Args:
        initial_value (float): Portföljens startvärde.
        mean_return (float): Årlig medelavkastning (t.ex. 0.07 för 7%).
        volatility (float): Årlig volatilitet (t.ex. 0.2 för 20%).
        days (int): Antal handelsdagar att simulera (standard 252).
        simulations (int): Antal simuleringar att köra.
    
    Returns:
        float: Förväntat slutvärde baserat på simuleringarna.
    """
    try:
        results = []
        for _ in range(simulations):
            # Generera dagliga avkastningar baserat på normalfördelning
            daily_returns = np.random.normal(mean_return / days, volatility / np.sqrt(days), days)
            # Beräkna prisutvecklingen: cumulativ produkt av (1 + avkastning)
            price_series = initial_value * (1 + daily_returns).cumprod()
            results.append(price_series[-1])
        expected_value = np.mean(results)
        logging.info("Normalfördelad Monte Carlo-simulering genomförd.")
        return expected_value
    except Exception as e:
        logging.error(f"Fel vid normal Monte Carlo-simulering: {str(e)}")
        return None

def monte_carlo_simulation_historical(returns: np.ndarray, num_simulations=1000, forecast_steps=252) -> list:
    """
    Monte Carlo-simulering baserad på historiska dagliga avkastningar.
    
    Args:
        returns (np.ndarray): Array med historiska dagliga avkastningar.
        num_simulations (int): Antal simuleringar att köra.
        forecast_steps (int): Antal dagar att simulera.
    
    Returns:
        list: Lista med simuleringar, där varje simulering är en tidsserie med framtida priser.
    """
    try:
        last_price = 100  # Exempelvärde, anpassa vid behov
        simulations = []
        for _ in range(num_simulations):
            price_series = [last_price]
            for _ in range(forecast_steps):
                simulated_return = np.random.choice(returns)
                new_price = price_series[-1] * (1 + simulated_return)
                price_series.append(new_price)
            simulations.append(price_series)
        logging.info("Historisk baserad Monte Carlo-simulering genomförd.")
        return simulations
    except Exception as e:
        logging.error(f"Fel vid historisk Monte Carlo-simulering: {str(e)}")
        return None

def ensemble_monte_carlo(initial_value, mean_return, volatility, historical_returns, days=252, simulations=1000, weight_normal=0.4, weight_historical=0.6):
    """
    Kombinerar resultaten från två Monte Carlo-simuleringar:
    - En baserad på normalfördelade avkastningar.
    - En baserad på historiska avkastningar.
    
    Resultaten kombineras med en viktad medelvärdesstrategi.
    
    Args:
        initial_value (float): Portföljens startvärde.
        mean_return (float): Årlig medelavkastning.
        volatility (float): Årlig volatilitet.
        historical_returns (np.ndarray): Array med historiska dagliga avkastningar.
        days (int): Antal dagar att simulera.
        simulations (int): Antal simuleringar att köra.
        weight_normal (float): Vikt för den normalfördelade simuleringen.
        weight_historical (float): Vikt för den historiska simuleringen.
    
    Returns:
        float: Ensemble-estimerat framtida portföljvärde.
    """
    # Kör normalfördelad simulering
    expected_value_normal = monte_carlo_simulation_normal(initial_value, mean_return, volatility, days, simulations)
    
    # Kör historisk baserad simulering
    historical_simulations = monte_carlo_simulation_historical(historical_returns, simulations, days)
    if historical_simulations is None:
        expected_value_historical = 0
    else:
        final_values = [series[-1] for series in historical_simulations]
        expected_value_historical = np.mean(final_values)
    
    # Kombinera resultaten med en viktad medelvärdesstrategi
    ensemble_value = weight_normal * expected_value_normal + weight_historical * expected_value_historical
    return ensemble_value

# Exempelanrop
if __name__ == "__main__":
    # Parametrar för normalfördelad simulering
    initial_value = 100000  # Exempel: 100 000 kronor
    mean_return = 0.07       # 7% årlig medelavkastning
    volatility = 0.2         # 20% årlig volatilitet
    
    # Exempel på historiska dagliga avkastningar (ersätt med riktiga data vid behov)
    historical_returns = np.array([0.001, -0.002, 0.003, 0.002, -0.001])
    
    # Kör ensemble-simuleringen
    ensemble_value = ensemble_monte_carlo(initial_value, mean_return, volatility, historical_returns)
    print("Ensemble beräknat framtida portföljvärde:", ensemble_value)