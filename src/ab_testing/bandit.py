"""
Multi-Armed Bandit for model selection.
Supports UCB1 (Upper Confidence Bound) and Epsilon-Greedy strategies.
Each "arm" is a recommendation model. Reward = user clicked / rated the recommendation.
"""
import math
import json
import random
from pathlib import Path
from typing import List, Dict, Optional


class ModelBandit:
    """
    Selects which recommendation model to use for each request,
    balancing exploration (trying all models) and exploitation (using the best one).

    UCB1:  score(arm) = avg_reward(arm) + sqrt(2 * ln(total_pulls) / pulls(arm))
    ε-greedy: with probability ε pick random arm, else pick best arm
    """

    def __init__(
        self,
        model_names: List[str],
        strategy: str = "ucb",
        epsilon: float = 0.1,
        state_path: Optional[Path] = None,
    ):
        assert strategy in ("ucb", "epsilon_greedy"), "strategy must be 'ucb' or 'epsilon_greedy'"
        self.strategy    = strategy
        self.epsilon     = epsilon
        self.state_path  = state_path
        self._arms: Dict[str, dict] = {
            name: {"pulls": 0, "total_reward": 0.0, "avg_reward": 0.0}
            for name in model_names
        }
        self._total_pulls = 0

        if state_path and Path(state_path).exists():
            self._load(state_path)

    # ── Selection ────────────────────────────────────────────────

    def select(self, user_id: Optional[int] = None) -> str:
        """Return the name of the model to use for this request."""
        # Ensure every arm is tried at least once
        for name, arm in self._arms.items():
            if arm["pulls"] == 0:
                return name

        if self.strategy == "ucb":
            return self._ucb_select()
        return self._epsilon_greedy_select()

    def _ucb_select(self) -> str:
        best, best_score = None, -1.0
        for name, arm in self._arms.items():
            bonus = math.sqrt(2 * math.log(self._total_pulls) / arm["pulls"])
            score = arm["avg_reward"] + bonus
            if score > best_score:
                best, best_score = name, score
        return best

    def _epsilon_greedy_select(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(list(self._arms.keys()))
        return max(self._arms, key=lambda n: self._arms[n]["avg_reward"])

    # ── Feedback ─────────────────────────────────────────────────

    def update(self, model_name: str, reward: float) -> None:
        """
        Update arm statistics after observing reward.
        reward: 1.0 = user interacted (click/rating), 0.0 = no interaction.
        """
        if model_name not in self._arms:
            return
        arm = self._arms[model_name]
        arm["pulls"]        += 1
        arm["total_reward"] += reward
        arm["avg_reward"]    = arm["total_reward"] / arm["pulls"]
        self._total_pulls   += 1

        if self.state_path:
            self._save(self.state_path)

    # ── Reporting ─────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "strategy":    self.strategy,
            "total_pulls": self._total_pulls,
            "arms": {
                name: {
                    "pulls":      arm["pulls"],
                    "avg_reward": round(arm["avg_reward"], 4),
                    "total_reward": round(arm["total_reward"], 2),
                }
                for name, arm in self._arms.items()
            },
            "current_best": max(self._arms, key=lambda n: self._arms[n]["avg_reward"])
                            if self._total_pulls > 0 else None,
        }

    # ── Persistence ───────────────────────────────────────────────

    def _save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump({"arms": self._arms, "total_pulls": self._total_pulls}, f, indent=2)

    def _load(self, path: Path) -> None:
        with open(path) as f:
            data = json.load(f)
        self._arms        = data["arms"]
        self._total_pulls = data["total_pulls"]
        print(f"Bandit: loaded state from {path} ({self._total_pulls} pulls)")
