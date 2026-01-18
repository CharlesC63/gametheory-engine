"""
Core Abstract Interfaces for GameTheory Engine

This module defines the abstract base classes that all game implementations
must follow, ensuring a unified interface across Chess, Shogi, Poker, and Go.

Key Concepts:
- GameState: Represents a position in the game
- InformationSet: Groups states that are indistinguishable to a player (key for imperfect info games)
- Action: A legal move in the game
- Strategy: A probability distribution over actions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Generic, TypeVar
from enum import Enum
import numpy as np

# Type variables for generic implementations
S = TypeVar('S', bound='GameState')
A = TypeVar('A')  # Action type


class GameType(Enum):
    """Classification of game types by information structure."""
    PERFECT_INFO = "perfect_info"           # Chess, Go, Shogi
    IMPERFECT_INFO = "imperfect_info"       # Poker
    SIMULTANEOUS = "simultaneous"            # Rock-Paper-Scissors
    STOCHASTIC = "stochastic"               # Backgammon, Poker


class Player(Enum):
    """Player identifiers."""
    PLAYER_1 = 0
    PLAYER_2 = 1
    CHANCE = -1  # For stochastic games (dealing cards, dice)


@dataclass
class GameConfig:
    """Configuration parameters for a game."""
    name: str
    game_type: GameType
    num_players: int
    max_actions: int
    is_zero_sum: bool = True
    has_chance_nodes: bool = False
    
    
class GameState(ABC, Generic[A]):
    """
    Abstract base class for game states.
    
    A GameState represents a complete description of the game at a point in time.
    For perfect information games, this is everything.
    For imperfect information games, this includes hidden information.
    """
    
    @abstractmethod
    def get_current_player(self) -> Player:
        """Return the player to act at this state."""
        pass
    
    @abstractmethod
    def get_legal_actions(self) -> List[A]:
        """Return list of legal actions from this state."""
        pass
    
    @abstractmethod
    def apply_action(self, action: A) -> 'GameState[A]':
        """Apply action and return new state. Does not modify self."""
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if this is a terminal (game-over) state."""
        pass
    
    @abstractmethod
    def get_payoffs(self) -> Dict[Player, float]:
        """
        Return payoffs for each player at a terminal state.
        Should only be called when is_terminal() is True.
        """
        pass
    
    @abstractmethod
    def get_information_set_key(self, player: Player) -> str:
        """
        Return a unique string key identifying the information set
        for the given player at this state.
        
        For perfect information games, this is just the state itself.
        For imperfect information games, this groups states that
        are indistinguishable to the player.
        """
        pass
    
    @abstractmethod
    def clone(self) -> 'GameState[A]':
        """Return a deep copy of this state."""
        pass


@dataclass
class InformationSet:
    """
    An information set groups game states that are indistinguishable
    to a particular player.
    
    In poker: your cards + board + betting history (but not opponent's cards)
    In chess: the entire board (perfect information, so |info_set| = 1)
    
    This is the fundamental unit for strategy computation in CFR.
    """
    key: str
    player: Player
    legal_actions: List[Any]
    
    # Strategy data (accumulated through CFR iterations)
    regret_sum: np.ndarray = field(default_factory=lambda: np.array([]))
    strategy_sum: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        n_actions = len(self.legal_actions)
        if len(self.regret_sum) == 0:
            self.regret_sum = np.zeros(n_actions)
        if len(self.strategy_sum) == 0:
            self.strategy_sum = np.zeros(n_actions)
    
    def get_strategy(self) -> np.ndarray:
        """
        Compute current strategy via regret matching.
        
        This is the core of CFR: actions with higher regret
        get higher probability.
        """
        positive_regrets = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(positive_regrets)
        
        if normalizing_sum > 0:
            return positive_regrets / normalizing_sum
        else:
            # Uniform distribution if no positive regrets
            return np.ones(len(self.legal_actions)) / len(self.legal_actions)
    
    def get_average_strategy(self) -> np.ndarray:
        """
        Compute average strategy over all iterations.
        
        This converges to Nash equilibrium in two-player zero-sum games.
        """
        normalizing_sum = np.sum(self.strategy_sum)
        
        if normalizing_sum > 0:
            return self.strategy_sum / normalizing_sum
        else:
            return np.ones(len(self.legal_actions)) / len(self.legal_actions)
    
    def update_regrets(self, action_regrets: np.ndarray):
        """Add regrets for this iteration."""
        self.regret_sum += action_regrets
    
    def update_strategy_sum(self, reach_probability: float, strategy: np.ndarray):
        """Accumulate strategy weighted by reach probability."""
        self.strategy_sum += reach_probability * strategy


class Game(ABC, Generic[S, A]):
    """
    Abstract base class for games.
    
    Provides the interface for creating initial states and
    querying game properties.
    """
    
    @abstractmethod
    def get_config(self) -> GameConfig:
        """Return game configuration."""
        pass
    
    @abstractmethod
    def get_initial_state(self) -> S:
        """Return the starting state of the game."""
        pass
    
    @abstractmethod
    def get_num_players(self) -> int:
        """Return number of players (excluding chance)."""
        pass
    
    def is_perfect_information(self) -> bool:
        """Return True if game has perfect information."""
        return self.get_config().game_type == GameType.PERFECT_INFO


class Solver(ABC, Generic[S, A]):
    """
    Abstract base class for game solvers.
    
    Solvers compute optimal or approximately optimal strategies
    for games.
    """
    
    @abstractmethod
    def solve(self, game: Game[S, A], **kwargs) -> Dict[str, InformationSet]:
        """
        Solve the game and return strategy for each information set.
        
        Returns:
            Dictionary mapping information set keys to InformationSet objects
            containing the computed strategies.
        """
        pass
    
    @abstractmethod
    def get_exploitability(self, game: Game[S, A], strategy: Dict[str, InformationSet]) -> float:
        """
        Compute how exploitable a strategy is.
        
        Returns:
            Exploitability in units of game value (e.g., big blinds in poker).
            A Nash equilibrium has exploitability = 0.
        """
        pass


# ==================== Finance Translation Interfaces ====================

@dataclass
class FinanceAnalog:
    """
    Maps game-theoretic concepts to finance applications.
    
    This is the "killer feature" that demonstrates understanding
    of how game theory applies to quantitative trading.
    """
    game_concept: str
    finance_concept: str
    description: str
    example: str


class FinanceTranslator(ABC):
    """
    Abstract interface for translating game strategies
    to financial applications.
    """
    
    @abstractmethod
    def translate_strategy(self, strategy: Dict[str, InformationSet]) -> Dict[str, Any]:
        """
        Convert a game strategy to financial terms.
        
        Example: Poker bluffing frequencies → Options overwriting frequencies
        """
        pass
    
    @abstractmethod
    def get_analogs(self) -> List[FinanceAnalog]:
        """Return list of game-to-finance concept mappings."""
        pass


# ==================== Utility Functions ====================

def compute_nash_distance(strategy1: Dict[str, np.ndarray], 
                          strategy2: Dict[str, np.ndarray]) -> float:
    """
    Compute the L2 distance between two strategies.
    Useful for measuring convergence.
    """
    total_distance = 0.0
    for key in strategy1:
        if key in strategy2:
            diff = strategy1[key] - strategy2[key]
            total_distance += np.sum(diff ** 2)
    return np.sqrt(total_distance)


def entropy(distribution: np.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    
    Higher entropy = more uncertainty = harder to exploit.
    """
    # Avoid log(0)
    dist = np.clip(distribution, 1e-10, 1.0)
    return -np.sum(dist * np.log2(dist))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL divergence D(P || Q).
    
    Measures how different P is from Q.
    Useful for measuring strategy deviation from GTO.
    """
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log2(p / q))
