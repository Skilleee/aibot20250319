"""
backtest_rl.py

Backtestar en redan tränad RL-modell (ex. PPO) mot historisk data,
räknar ut PnL, etc.
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from ai_learning.reinforcement_learning import TradingEnv

def backtest_rl_agent(df: pd.DataFrame, model_path="rl_trading_model.zip"):
    """
    df: DataFrame med kolumner som "close", "momentum", "volume"
    model_path: Var den tränade RL-modellen ligger
    Returnerar total PnL eller liknande.
    """
    # Ladda agenten
    model = PPO.load(model_path)
    env = TradingEnv(df=df)  # Samma environment
    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    final_value = env._get_portfolio_value()
    print(f"Backtest: Totalt reward = {total_reward:.2f}, slutligt portföljvärde = {final_value:.2f}")
    return final_value

if __name__ == "__main__":
    # Exempel: Ladda en CSV
    df = pd.DataFrame({
        "close": [100, 101, 102, 103, 104],
        "momentum": [0.1, 0.2, -0.1, 0.3, -0.2],
        "volume": [500, 400, 600, 550, 500],
    })

    backtest_rl_agent(df, model_path="rl_trading_model.zip")
