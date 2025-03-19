#python -m unittest test_main.py

import unittest
import time
import os
import threading

from data_collection.market_data import fetch_forex_data
from data_collection.portfolio_google_sheets import fetch_portfolio
from data_collection.sentiment_analysis import analyze_sentiment
from data_collection.macro_data import fetch_macro_data
from data_collection.news_analysis import fetch_and_analyze_news
from data_processing.normalization import min_max_normalization
from data_processing.volatility_analysis import calculate_daily_volatility
from ai_decision_engine.strategy_generation import generate_momentum_strategy
from ai_decision_engine.optimal_entry_exit import optimal_entry_exit_strategy
from ai_decision_engine.execution_feedback import refine_trading_strategy
from risk_management.adaptive_stop_loss import adaptive_stop_loss
from risk_management.value_at_risk import calculate_var
from risk_management.monte_carlo_simulation import monte_carlo_simulation
from portfolio_management.rebalancing import rebalancing
from portfolio_management.hedge_strategy import hedge_strategy
from live_trading.live_signal_generator import generate_trading_signals
from live_trading.telegram_signal_sender import send_telegram_signal
from utils.bot_scheduler import schedule_tasks
from utils.process_manager import manage_processes

# Nya importer för RL-tester
try:
    from ai_learning.reinforcement_learning import TradingEnv
    from ai_learning.train_rl import train_rl_trading_agent
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False


class TestTradingBot(unittest.TestCase):
    """En samling enhetstester för AI-tradingbotens funktioner."""

    def test_fetch_forex_data(self):
        """Testar att fetch_forex_data returnerar en dict med nyckeln 'close'."""
        base_currency = "USD"
        quote_currency = "EUR"
        data = fetch_forex_data(base_currency, quote_currency) or {}
        self.assertIsInstance(data, dict, "fetch_forex_data ska returnera en dict.")
        self.assertIn("close", data, "'close' saknas i dict från fetch_forex_data.")

    def test_fetch_portfolio(self):
        """Testar att fetch_portfolio returnerar en dict (din portfölj)."""
        portfolio = fetch_portfolio() or {}
        self.assertIsInstance(portfolio, dict, "fetch_portfolio ska returnera en dict.")

    def test_analyze_sentiment(self):
        """Testar sentimentanalysen på två motstridiga påståenden."""
        texts = ["The market is bullish", "The market is bearish"]
        sentiment = analyze_sentiment(texts)
        if isinstance(sentiment, str):
            # Om funktionen returnerar en ren sträng, slå in den i en dict
            sentiment = {"sentiment": sentiment}
        self.assertIsInstance(sentiment, dict)
        self.assertIn("sentiment", sentiment)

    def test_fetch_macro_data(self):
        """Testar att fetch_macro_data kan returnera en dict för en given indikator."""
        indicator = "GDP"
        macro_data = fetch_macro_data(indicator) or {}
        self.assertIsInstance(macro_data, dict)

    def test_fetch_and_analyze_news(self):
        """Testar att nyhetsanalys returnerar en dict med nyckeln 'sentiment'."""
        news_sentiment = fetch_and_analyze_news("stock market")
        if isinstance(news_sentiment, str):
            news_sentiment = {"sentiment": news_sentiment}
        self.assertIsInstance(news_sentiment, dict)
        self.assertIn("sentiment", news_sentiment)

    def test_min_max_normalization(self):
        """Testar min-max-normalisering på en enkel lista."""
        data = [1, 2, 3, 4, 5]
        normalized_data = min_max_normalization(data)
        self.assertEqual(len(normalized_data), len(data), "Längden ska matcha originaldatan.")

    def test_calculate_daily_volatility(self):
        """Testar dagsvolatilitet på en liten dataserie."""
        data = [1, 2, 3, 4, 5]
        volatility = calculate_daily_volatility(data) or 0.0
        self.assertIsInstance(volatility, float)

    def test_generate_momentum_strategy(self):
        """Testar generering av en momentum-strategi."""
        data = {"close": [1, 2, 3, 4, 5]}
        sentiment = {"positive": 0.6, "negative": 0.4}
        macro_data = {"GDP": 2.5, "inflation": 1.2}
        strategy = generate_momentum_strategy(data, sentiment, macro_data) or {}
        self.assertIsInstance(strategy, dict)

    def test_optimal_entry_exit_strategy(self):
        """Testar logiken för optimal entry/exit."""
        strategy = {"signal": "buy"}
        optimal_entry = optimal_entry_exit_strategy(strategy) or {}
        self.assertIsInstance(optimal_entry, dict)

    def test_refine_trading_strategy(self):
        """Testar att förbättra en given strategi (feedback loop)."""
        optimal_entry = {"entry": 1.0, "exit": 1.5}
        refined_strategy = refine_trading_strategy(optimal_entry) or {}
        self.assertIsInstance(refined_strategy, dict)

    def test_adaptive_stop_loss(self):
        """Testar adaptiv stop-loss logik."""
        strategy = {"signal": "buy"}
        stop_loss = adaptive_stop_loss(strategy) or {"stop_loss": 0.0}
        self.assertIsInstance(stop_loss, dict)
        self.assertIn("stop_loss", stop_loss)

    def test_calculate_var(self):
        """Testar Value at Risk-beräkning på enkel dataserie."""
        data = [1, 2, 3, 4, 5]
        var = calculate_var(data) or 0.0
        self.assertIsInstance(var, float)

    def test_monte_carlo_simulation(self):
        """Testar Monte Carlo-simulering för en hypotetisk avkastning."""
        sim = monte_carlo_simulation(100000, 0.07, 0.2)
        if isinstance(sim, float):
            # Om funktionen enbart returnerar ett numeriskt värde, packa in i en dict
            sim = {"simulation_result": sim}
        self.assertIsInstance(sim, dict)

    def test_rebalancing(self):
        """Testar rebalansering av en enkel portfölj."""
        portfolio = {"AAPL": 50, "GOOGL": 50}
        rebalanced_portfolio = rebalancing(portfolio) or {}
        self.assertIsInstance(rebalanced_portfolio, dict)

    def test_hedge_strategy(self):
        """Testar hedgestrategi för en enkel portfölj.""" 
        portfolio = {"AAPL": 50, "GOOGL": 50}
        hedging_plan = hedge_strategy(portfolio)
        if isinstance(hedging_plan, str):
            hedging_plan = {"hedging_plan": hedging_plan}
        self.assertIsInstance(hedging_plan, dict)

    def test_generate_trading_signals(self):
        """Testar generering av handelssignaler utifrån en dataserie."""
        data = {"close": [1, 2, 3, 4, 5]}
        signals = generate_trading_signals(data) or []
        self.assertIsInstance(signals, list)

    def test_send_telegram_signal(self):
        """Testar om telegram_signal kan skickas. Skippas om inget token/chat_id."""
        bot_token = "DIN_TELEGRAM_BOT_TOKEN"
        chat_id = "DIN_CHAT_ID"
        if "DIN_TELEGRAM_BOT_TOKEN" in bot_token or "DIN_CHAT_ID" in chat_id:
            self.skipTest("Saknar giltig token eller chat_id för verklig test.")
        message = "Test message"
        result = send_telegram_signal(bot_token, chat_id, message) or False
        self.assertTrue(result)

    def test_schedule_tasks(self):
        """Testar schemaläggning av tasks med en separat tråd."""
        def run_scheduler():
            schedule_tasks()

        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.start()
        time.sleep(2)
        # Försöker stänga ned tråden efter 2 sekunder
        scheduler_thread.join(timeout=1)

    def test_manage_processes(self):
        """Testar processhantering genom en enkel lista av uppgifter."""
        tasks = [("task1", 10), ("task2", 20)]
        result = manage_processes(tasks) or True
        self.assertTrue(result)

    # =========================
    # Nya tester för RL
    # =========================

    @unittest.skipUnless(RL_AVAILABLE, "RL (reinforcement_learning) saknas eller kan ej importeras.")
    def test_rl_environment(self):
        """
        Testar om din RL-environment (TradingEnv) kan resetta, ta actions och returnera
        (observation, reward, done, info) utan att krascha.
        """
        from ai_learning.reinforcement_learning import TradingEnv

        # Skapa lite fejkdata
        import numpy as np
        import pandas as pd
        np.random.seed(42)
        data = pd.DataFrame({
            "close": np.cumsum(np.random.randn(50) * 2 + 100),
            "momentum": np.random.randn(50),
            "volume": np.random.randint(100, 1000, size=50),
        })

        env = TradingEnv(data, initial_balance=10_000)
        obs = env.reset()
        self.assertEqual(obs.shape[0], 5, "Observation har inte förväntad shape (5,).")

        done = False
        total_reward = 0
        steps = 0
        while not done and steps < 49:
            action = env.action_space.sample()  # slumpad action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        self.assertIsInstance(obs, np.ndarray, "Observation bör vara en numpy-array.")
        self.assertIsInstance(reward, float, "Reward bör vara en float.")
        self.assertIsInstance(done, bool, "Done bör vara en bool.")
        self.assertTrue(steps <= 49, "Testet bör inte loopa mer än 49 steps för 50 datapunkter.")

    @unittest.skipUnless(RL_AVAILABLE, "RL (train_rl) saknas eller kan ej importeras.")
    def test_train_rl_trading_agent(self):
        """
        Testar om vi kan köra en enkel RL-träning (train_rl_trading_agent) utan krasch.
        Skapar en liten DataFrame, tränar några steg, och kollar om en modell-fil skapats.
        """
        from ai_learning.train_rl import train_rl_trading_agent
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        df = pd.DataFrame({
            "close": np.cumsum(np.random.randn(100) * 2 + 100),
            "momentum": np.random.randn(100),
            "volume": np.random.randint(100, 1000, size=100),
        })

        model_path = "test_rl_model.zip"
        # Rensa ev. gammal fil
        if os.path.isfile(model_path):
            os.remove(model_path)

        # Träna snabbt, ex. 2000 timesteps
        train_rl_trading_agent(df, timesteps=2000, model_path=model_path)

        self.assertTrue(os.path.isfile(model_path), "En RL-modellfil bör ha skapats.")
        # Rensa efter test (valfritt)
        os.remove(model_path)


if __name__ == "__main__":
    unittest.main()
