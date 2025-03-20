"""
train_rl.py

Tränar en RL-agent (ex. PPO) med hjälp av TradingEnv och stable_baselines3.
Sparar den tränade modellen till en fil.
"""

import os
import logging
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ai_learning.reinforcement_learning import TradingEnv

logging.basicConfig(
    filename="train_rl.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_rl_trading_agent(df: pd.DataFrame, timesteps: int = 100_000, model_path: str = "rl_trading_model.zip"):
    """
    Tränar en RL-agent med PPO-algoritmen på TradingEnv.
    
    Args:
        df (pd.DataFrame): DataFrame med kolumnerna "close", "momentum", "volume" (och eventuellt "return").
        timesteps (int): Antal timesteps att träna RL-agenten.
        model_path (str): Sökväg för att spara den tränade modellen.
        
    Returns:
        PPO: Den tränade modellen.
    """
    def make_env():
        return TradingEnv(df=df)

    # Skapa en vektoriserad miljö med DummyVecEnv
    env = DummyVecEnv([make_env])
    
    # Initiera PPO-modellen
    model = PPO("MlpPolicy", env, verbose=1)
    logging.info(f"Startar RL-träning i {timesteps} steg...")
    
    # Träna modellen
    model.learn(total_timesteps=timesteps)
    
    # Spara den tränade modellen
    model.save(model_path)
    logging.info(f"✅ RL-modell tränad och sparad till {model_path}")
    
    # Stäng miljön för att frigöra resurser
    env.close()
    
    return model

if __name__ == "__main__":
    # Exempel: Ladda historisk data
    # Se till att data har rätt kolumnnamn (t.ex. "close", "momentum", "volume")
    df = pd.DataFrame({
        "close": [101, 102, 103, 104, 105],
        "momentum": [0.1, -0.2, 0.05, 0.3, -0.1],
        "volume": [500, 600, 550, 700, 650],
    })

    train_rl_trading_agent(df, timesteps=10_000, model_path="rl_trading_model.zip")
