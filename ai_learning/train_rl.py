"""
train_rl.py
Tränar en RL-agent (ex. PPO) med hjälp av TradingEnv och stabil_baselines3.
Sparar den tränade modellen till en fil.
"""

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ai_learning.reinforcement_learning import TradingEnv

def train_rl_trading_agent(df: pd.DataFrame, timesteps=100_000, model_path="rl_trading_model.zip"):
    """
    df: DataFrame med minst kolumnen "Close" (och ev. "Sentiment", etc.)
    timesteps: Antal träningssteg
    model_path: Var vi sparar modellen
    """
    # Skapa vårt gym-env
    def make_env():
        return TradingEnv(df=df)

    env = DummyVecEnv([make_env])

    # Välj en RL-algoritm, ex. PPO, DQN, A2C, etc.
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    print(f"✅ RL-modell tränad och sparad till {model_path}")

if __name__ == "__main__":
    # Exempelkod: ladda historisk data (Close, Sentiment, etc.)
    # Du kan hämta via data_collection, ex. fetch_forex_data med en tidsserie, sen preppa en df
    df = pd.read_csv("historical_data.csv")
    # ev. df["Sentiment"] = ...
    train_rl_trading_agent(df, timesteps=20_000, model_path="rl_trading_model.zip")
