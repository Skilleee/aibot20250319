import logging
import gym
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3 import PPO

# Konfigurera loggning
logging.basicConfig(
    filename="reinforcement_learning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class TradingEnv(gym.Env):
    """
    En förbättrad förstärkningsinlärningsmiljö för trading.
    
    Denna miljö hanterar en portfölj med ett initialt kapital (balance) och
    en mängd innehav (holding) av en tillgång. Handlingsutrymmet:
        0: HOLD (inga förändringar)
        1: BUY (köp med 10% av aktuell balans)
        2: SELL (sälj alla innehav)
    
    Observationen är en vektor: [close, momentum, volume, balance, holding].
    Belöningen definieras som förändringen i portföljvärde.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10_000):
        """
        Initierar miljön med historiska data och startkapital.
        
        Args:
            data (pd.DataFrame): DataFrame med kolumnerna ["close", "momentum", "volume", "return"].
            initial_balance (float): Startkapital, t.ex. USD.
        """
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance

        # Definiera action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        # Definiera observation space: [close, momentum, volume, balance, holding]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        # Interna tillstånd
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.holding = 0.0

        # För reproducibilitet
        self.seed()

    def seed(self, seed=None):
        """
        Sätter seed för slumpgenereringen.
        
        Returns:
            list: Seed som används.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _get_obs(self) -> np.ndarray:
        """
        Hämtar observationen: [close, momentum, volume, balance, holding].
        
        Returns:
            np.ndarray: Nuvarande observation.
        """
        row = self.data.iloc[self.current_step]
        obs = np.array([
            row["close"],
            row["momentum"],
            row["volume"],
            self.balance,
            self.holding
        ], dtype=np.float32)
        return obs

    def reset(self) -> np.ndarray:
        """
        Återställer miljön till startläge och returnerar den initiala observationen.
        
        Returns:
            np.ndarray: Initial observation.
        """
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.holding = 0.0
        logging.info("Miljön återställd: balance=%.2f, holding=%.2f", self.balance, self.holding)
        return self._get_obs()

    def step(self, action: int):
        """
        Utför en given handling och returnerar (observation, reward, done, info).
        
        Args:
            action (int): 0 (HOLD), 1 (BUY) eller 2 (SELL).
            
        Returns:
            tuple: (obs, reward, done, info)
        """
        # Spara nuvarande portföljvärde för rewardberäkning
        old_value = self._get_portfolio_value()

        # Utför handling
        self._take_action(action)
        logging.debug("Utförd handling: %d vid steg %d", action, self.current_step)

        # Gå till nästa steg
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Beräkna reward baserat på förändring i portföljvärde
        new_value = self._get_portfolio_value()
        reward = new_value - old_value
        logging.info("Steg: %d, Action: %d, Reward: %.2f, Portföljvärde: %.2f",
                     self.current_step, action, reward, new_value)

        # Hämta nästa observation
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return obs, reward, done, info

    def _take_action(self, action: int):
        """
        Utför en förenklad köp/sälj-logik.
        
        BUY: Investera 10% av nuvarande balans.
        SELL: Sälj alla innehav.
        HOLD: Gör ingenting.
        """
        price = self.data.iloc[self.current_step]["close"]
        if action == 1:  # BUY
            amount_to_spend = 0.1 * self.balance
            if amount_to_spend > 0:
                units = amount_to_spend / price
                self.holding += units
                self.balance -= amount_to_spend
                logging.debug("Köpte: %.4f enheter till pris %.2f", units, price)
        elif action == 2:  # SELL
            if self.holding > 0:
                self.balance += self.holding * price
                logging.debug("Sålde: %.4f enheter till pris %.2f", self.holding, price)
                self.holding = 0.0
        # Om action == 0 (HOLD) görs inget

    def _get_portfolio_value(self) -> float:
        """
        Beräknar nuvarande portföljvärde: balance + holding * current price.
        
        Returns:
            float: Nuvarande portföljvärde.
        """
        price = self.data.iloc[self.current_step]["close"]
        return self.balance + self.holding * price

    def render(self, mode="human"):
        """
        Render-funktion för att visa aktuell status för portföljen.
        """
        portfolio_value = self._get_portfolio_value()
        print(f"Steg: {self.current_step} | Balance: {self.balance:.2f} | Holding: {self.holding:.4f} | Portföljvärde: {portfolio_value:.2f}")

    def close(self):
        """
        Stänger miljön och frigör resurser vid behov.
        """
        logging.info("Miljön stängs.")

if __name__ == "__main__":
    # Exempel: Generera simulerad data
    np.random.seed(42)
    data = pd.DataFrame({
        "close": np.cumsum(np.random.randn(1000) * 2 + 100),
        "momentum": np.random.randn(1000),
        "volume": np.random.randint(100, 1000, size=1000),
        "return": np.random.randn(1000) / 100,
    })

    env = TradingEnv(data, initial_balance=10_000)

    # Träna en PPO-modell med Stable Baselines3
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)
    model.save("rl_trading_model.zip")
    print("📢 RL-modell är tränad och sparad!")
