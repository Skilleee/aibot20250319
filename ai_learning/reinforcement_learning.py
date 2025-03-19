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
    En utökad förstärkningsinlärningsmiljö för trading, baserad på
    ditt tidigare exempel. Den hanterar nu en 'holding' (antal enheter)
    och en 'balance' (t.ex. USD). Action = 0=HOLD, 1=BUY, 2=SELL.
    Reward = förändring i portföljvärde (balance + holding*price).
    """

    def __init__(self, data, initial_balance=10_000):
        """
        data: Pandas DataFrame med kolumner som ["close", "momentum", "volume", "return"].
        initial_balance: Startkapital i t.ex. USD.
        """
        super(TradingEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.initial_balance = initial_balance

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Exempel: observation med [pris, momentum, volume, balance, holding]
        # Du kan lägga till "return" eller ta bort "volume" etc. efter behov.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        # Interna variabler (balance, holding, etc.)
        self.reset()

    def _get_obs(self):
        """
        Hämtar observationen: [close, momentum, volume, balance, holding].
        """
        row = self.data.iloc[self.current_step]
        close = row["close"]
        momentum = row["momentum"]
        volume = row["volume"]

        obs = np.array([
            close,
            momentum,
            volume,
            self.balance,
            self.holding
        ], dtype=np.float32)

        return obs

    def reset(self):
        """
        Återställ miljön till startläge.
        """
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.holding = 0.0  # hur många "units" vi äger av instrumentet
        return self._get_obs()

    def step(self, action):
        """
        Utför en handling (0=HOLD, 1=BUY, 2=SELL).
        Returnerar (obs, reward, done, info).
        """
        # Spara gamla portföljvärdet för reward-beräkning
        old_value = self._get_portfolio_value()

        # Gör action
        self._take_action(action)

        self.current_step += 1
        done = (self.current_step >= len(self.data) - 1)

        # Räkna ut reward som skillnad i portföljvärde
        new_value = self._get_portfolio_value()
        reward = new_value - old_value

        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape)
        info = {}

        return obs, reward, done, info

    def _take_action(self, action):
        """
        En förenklad köp/sälj-logik.
        - BUY = investera 10% av balance
        - SELL = sälj allt vi äger
        - HOLD = gör ingenting
        """
        # Hämta priset på nuvarande step
        price = self.data.iloc[self.current_step]["close"]

        if action == 1:  # BUY
            amount_to_spend = 0.1 * self.balance
            units = amount_to_spend / price
            self.holding += units
            self.balance -= amount_to_spend

        elif action == 2:  # SELL
            # sälj all holding
            self.balance += self.holding * price
            self.holding = 0.0

        # HOLD => gör ingenting

    def _get_portfolio_value(self):
        """
        Nuvarande portföljvärde = balance + holding * price
        """
        price = self.data.iloc[self.current_step]["close"]
        return self.balance + self.holding * price


if __name__ == "__main__":
    # Exempel: Generera simulerad data
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "close": np.cumsum(np.random.randn(1000) * 2 + 100),
            "momentum": np.random.randn(1000),
            "volume": np.random.randint(100, 1000, size=1000),
            "return": np.random.randn(1000) / 100,
        }
    )

    env = TradingEnv(data, initial_balance=10_000)

    # Träna en PPO-modell
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    model.save("rl_trading_model.zip")
    print("📢 RL-modell är tränad och sparad!")
