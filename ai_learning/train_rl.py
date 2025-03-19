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

def train_rl_trading_agent(df: pd.DataFrame, timesteps=100_000, model_path="rl_trading_model.zip"):
    """
    df: DataFrame med kolumner "close", "momentum", "volume", etc.
    timesteps: antal timesteps att träna RL-agenten
    model_path: filnamn för att spara den tränade modellen
    """

    def make_env():
        return TradingEnv(df=df)

    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=1)
    logging.info(f"Startar RL-träning i {timesteps} steg...")

    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    logging.info(f"✅ RL-modell tränad och sparad till {model_path}")

if __name__ == "__main__":
    # Exempel: Ladda historisk data
    # (Byt ut mot riktig data. Du kan t.ex. hämta via fetch_forex_data eller fetch_multiple_stocks och göra en DataFrame.)
    df = pd.DataFrame({
        "close": [101, 102, 103, 104, 105],
        "momentum": [0.1, -0.2, 0.05, 0.3, -0.1],
        "volume": [500, 600, 550, 700, 650],
    })

    train_rl_trading_agent(df, timesteps=10_000, model_path="rl_trading_model.zip")
