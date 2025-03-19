import logging

import gym
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3 import PPO

# Konfigurera loggning
logging.basicConfig(filename="reinforcement_learning.log", level=logging.INFO)


class TradingEnv(gym.Env):
    """
    Anpassad förstärkningsinlärningsmiljö för trading.
    """

    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0

        # Definiera action space: 0 = HOLD, 1 = BUY, 2 = SELL
        self.action_space = spaces.Discrete(3)

        # Definiera observationsutrymmet: Pris, momentum, volym
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

    def reset(self):
        """Återställer miljön till startläge."""
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        """Returnerar nästa observation."""
        obs = self.data.iloc[self.current_step][["close", "momentum", "volume"]].values
        return obs.astype(np.float32)

    def step(self, action):
        """Utför en handling och returnerar observation, belöning och status."""
        self.current_step += 1

        done = self.current_step >= len(self.data) - 1
        reward = self._calculate_reward(action)

        return self._next_observation(), reward, done, {}

    def _calculate_reward(self, action):
        """Beräknar belöning baserat på om AI:n fattade rätt beslut."""
        if action == 1:  # BUY
            return self.data.iloc[self.current_step]["return"]
        elif action == 2:  # SELL
            return -self.data.iloc[self.current_step]["return"]
        return 0  # HOLD


# Träna RL-modellen
if __name__ == "__main__":
    # Simulerad prisdata
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "close": np.cumsum(np.random.randn(1000) * 2 + 100),
            "momentum": np.random.randn(1000),
            "volume": np.random.randint(100, 1000, size=1000),
            "return": np.random.randn(1000) / 100,
        }
    )

    env = TradingEnv(data)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    print("📢 Reinforcement Learning-modellen är tränad och redo att testas!")
