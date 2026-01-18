"""
Kuhn Poker Implementation with Counterfactual Regret Minimization (CFR)

Kuhn Poker is the simplest non-trivial poker game, making it ideal for
demonstrating CFR. Despite its simplicity, it captures the essential
elements of poker strategy: bluffing, value betting, and calling.

Game Rules:
- 3-card deck: Jack, Queen, King (J < Q < K)
- Each player antes 1 chip, gets 1 card
- Player 1 acts first: check or bet (1 chip)
- Player 2 responds: check/bet or fold/call
- If checked around or called, higher card wins

This implementation includes:
1. Complete game state representation
2. Vanilla CFR algorithm
3. Exploitability calculation
4. Nash equilibrium strategy

Finance Parallel:
- CFR's regret minimization mirrors adaptive market making
- Bluffing frequency maps to information asymmetry exploitation
- Nash equilibrium = GTO strategy (Game Theory Optimal)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.game_interface import (
    GameState, Game, Solver, InformationSet, GameConfig, 
    GameType, Player, FinanceAnalog, entropy
)


class Card(Enum):
    """Kuhn Poker cards: Jack < Queen < King"""
    JACK = 0
    QUEEN = 1
    KING = 2
    
    def __lt__(self, other):
        return self.value < other.value
    
    def __str__(self):
        return ['J', 'Q', 'K'][self.value]


class Action(Enum):
    """Possible actions in Kuhn Poker"""
    CHECK = 'c'   # Pass the action (risk-free)
    BET = 'b'     # Put 1 chip in
    FOLD = 'f'    # Give up the hand
    CALL = 'k'    # Match opponent's bet
    
    def __str__(self):
        return self.value


# Action sequences and their meanings
# Empty: Player 1 to act
# 'c': P1 checked, P2 to act
# 'b': P1 bet, P2 to act (fold/call)
# 'cc': Both checked -> showdown
# 'bc': P1 bet, P2 folded (invalid action)
# 'cb': P1 checked, P2 bet, P1 to act (fold/call)
# 'bb': P1 bet, P2 called -> showdown
# 'cbk': P1 check-called -> showdown
# 'cbf': P1 check-folded -> P2 wins


@dataclass
class KuhnState(GameState):
    """
    Complete game state for Kuhn Poker.
    
    Attributes:
        cards: Tuple of (P1 card, P2 card)
        history: String of actions taken (e.g., 'cb' = check, bet)
    """
    cards: Tuple[Card, Card]
    history: str = ""
    
    def get_current_player(self) -> Player:
        """Determine who acts based on history length and pattern."""
        if self.is_terminal():
            return Player.CHANCE
        
        # After P1 checks and P2 bets, P1 acts again
        if self.history == 'cb':
            return Player.PLAYER_1
        
        # P1 acts on even-length histories, P2 on odd
        return Player.PLAYER_1 if len(self.history) % 2 == 0 else Player.PLAYER_2
    
    def get_legal_actions(self) -> List[Action]:
        """Return legal actions based on betting history."""
        if self.is_terminal():
            return []
        
        # Facing a bet: can fold or call
        if self.history.endswith('b'):
            return [Action.FOLD, Action.CALL]
        
        # No bet yet: can check or bet
        return [Action.CHECK, Action.BET]
    
    def apply_action(self, action: Action) -> 'KuhnState':
        """Apply action and return new state."""
        return KuhnState(
            cards=self.cards,
            history=self.history + action.value
        )
    
    def is_terminal(self) -> bool:
        """Check if game is over."""
        h = self.history
        # Game ends on:
        # - 'cc': check-check (showdown)
        # - 'bf': bet-fold (P1 wins)
        # - 'bk': bet-call (showdown)
        # - 'cbf': check-bet-fold (P2 wins)
        # - 'cbk': check-bet-call (showdown)
        return h in ['cc', 'bf', 'bk', 'cbf', 'cbk']
    
    def get_payoffs(self) -> Dict[Player, float]:
        """
        Calculate payoffs at terminal state.
        
        Pot structure:
        - Both ante 1, so base pot = 2
        - If bet was called, pot = 4
        
        Returns payoff relative to initial ante (so break-even = 0).
        """
        h = self.history
        p1_card, p2_card = self.cards
        
        # Player 2 folded to P1's bet
        if h == 'bf':
            return {Player.PLAYER_1: 1, Player.PLAYER_2: -1}  # P1 wins P2's ante
        
        # Player 1 folded to P2's bet (after P1 checked)
        if h == 'cbf':
            return {Player.PLAYER_1: -1, Player.PLAYER_2: 1}  # P2 wins P1's ante
        
        # Showdown - determine winner
        p1_wins = p1_card > p2_card
        
        # Check-check: winner takes 1 chip (the other's ante)
        if h == 'cc':
            return {Player.PLAYER_1: 1, Player.PLAYER_2: -1} if p1_wins else {Player.PLAYER_1: -1, Player.PLAYER_2: 1}
        
        # Bet-call or check-bet-call: winner takes 2 chips
        if h in ['bk', 'cbk']:
            return {Player.PLAYER_1: 2, Player.PLAYER_2: -2} if p1_wins else {Player.PLAYER_1: -2, Player.PLAYER_2: 2}
        
        raise ValueError(f"Unknown terminal history: {h}")
    
    def get_information_set_key(self, player: Player) -> str:
        """
        Return information set key.
        
        In Kuhn Poker, a player knows:
        - Their own card
        - The betting history
        
        They do NOT know the opponent's card.
        """
        if player == Player.PLAYER_1:
            return f"{self.cards[0]}{self.history}"
        else:
            return f"{self.cards[1]}{self.history}"
    
    def clone(self) -> 'KuhnState':
        return KuhnState(cards=self.cards, history=self.history)
    
    def __str__(self) -> str:
        return f"Cards: {self.cards[0]},{self.cards[1]} | History: {self.history or 'start'}"


class KuhnPoker(Game[KuhnState, Action]):
    """Kuhn Poker game implementation."""
    
    def __init__(self):
        self._config = GameConfig(
            name="Kuhn Poker",
            game_type=GameType.IMPERFECT_INFO,
            num_players=2,
            max_actions=2,  # Check/Bet or Fold/Call
            is_zero_sum=True,
            has_chance_nodes=True
        )
    
    def get_config(self) -> GameConfig:
        return self._config
    
    def get_initial_state(self) -> KuhnState:
        """Deal random cards to start the game."""
        cards = random.sample(list(Card), 2)
        return KuhnState(cards=(cards[0], cards[1]))
    
    def get_all_initial_states(self) -> List[Tuple[KuhnState, float]]:
        """
        Return all possible deals with their probabilities.
        
        6 possible deals: JQ, JK, QJ, QK, KJ, KQ
        Each has probability 1/6.
        """
        states = []
        for c1 in Card:
            for c2 in Card:
                if c1 != c2:
                    states.append((KuhnState(cards=(c1, c2)), 1/6))
        return states
    
    def get_num_players(self) -> int:
        return 2


class KuhnCFR(Solver[KuhnState, Action]):
    """
    Counterfactual Regret Minimization solver for Kuhn Poker.
    
    CFR iteratively:
    1. Computes counterfactual values for each action
    2. Accumulates regret for not taking better actions
    3. Updates strategy proportionally to positive regrets
    
    Over many iterations, the average strategy converges to Nash equilibrium.
    
    Finance Parallel:
    - Regret minimization is analogous to adaptive trading strategies
    - Counterfactual reasoning = "what if I had hedged differently?"
    - Convergence to Nash = market efficiency
    """
    
    def __init__(self):
        self.info_sets: Dict[str, InformationSet] = {}
        self.iterations = 0
    
    def get_or_create_info_set(self, key: str, player: Player, 
                               legal_actions: List[Action]) -> InformationSet:
        """Get existing info set or create new one."""
        if key not in self.info_sets:
            self.info_sets[key] = InformationSet(
                key=key,
                player=player,
                legal_actions=legal_actions
            )
        return self.info_sets[key]
    
    def cfr(self, state: KuhnState, reach_probs: np.ndarray) -> np.ndarray:
        """
        Recursive CFR traversal.
        
        Args:
            state: Current game state
            reach_probs: Probability of reaching this state for each player
                        [P(P1 plays to reach here), P(P2 plays to reach here)]
        
        Returns:
            Expected utilities for each player from this state.
        """
        # Terminal state: return payoffs
        if state.is_terminal():
            payoffs = state.get_payoffs()
            return np.array([payoffs[Player.PLAYER_1], payoffs[Player.PLAYER_2]])
        
        player = state.get_current_player()
        player_idx = player.value
        opponent_idx = 1 - player_idx
        
        # Get information set
        info_key = state.get_information_set_key(player)
        legal_actions = state.get_legal_actions()
        info_set = self.get_or_create_info_set(info_key, player, legal_actions)
        
        # Get current strategy via regret matching
        strategy = info_set.get_strategy()
        
        # Compute counterfactual values for each action
        n_actions = len(legal_actions)
        action_values = np.zeros((n_actions, 2))
        
        for i, action in enumerate(legal_actions):
            next_state = state.apply_action(action)
            
            # Update reach probabilities
            new_reach = reach_probs.copy()
            new_reach[player_idx] *= strategy[i]
            
            action_values[i] = self.cfr(next_state, new_reach)
        
        # Expected values under current strategy
        expected_values = np.zeros(2)
        for i in range(n_actions):
            expected_values += strategy[i] * action_values[i]
        
        # Compute and accumulate regrets
        # Regret = (value of action) - (value of strategy we played)
        # Weighted by opponent's reach probability (counterfactual)
        counterfactual_reach = reach_probs[opponent_idx]
        
        action_regrets = np.zeros(n_actions)
        for i in range(n_actions):
            action_regrets[i] = counterfactual_reach * (action_values[i, player_idx] - expected_values[player_idx])
        
        info_set.update_regrets(action_regrets)
        
        # Accumulate strategy for computing average
        info_set.update_strategy_sum(reach_probs[player_idx], strategy)
        
        return expected_values
    
    def train(self, game: KuhnPoker, iterations: int, verbose: bool = True) -> Dict[str, InformationSet]:
        """
        Train CFR for specified iterations.
        
        Args:
            game: KuhnPoker instance
            iterations: Number of CFR iterations
            verbose: Print progress
        
        Returns:
            Trained information sets with converged strategies
        """
        expected_value = 0.0
        
        for i in range(iterations):
            # Go through all possible deals
            for initial_state, prob in game.get_all_initial_states():
                # Run CFR from this deal
                values = self.cfr(initial_state, np.array([1.0, 1.0]))
                expected_value += prob * values[0]
            
            self.iterations += 1
            
            if verbose and (i + 1) % (iterations // 10 or 1) == 0:
                exploitability = self.get_exploitability(game, self.info_sets)
                print(f"Iteration {i + 1}/{iterations} | "
                      f"Exploitability: {exploitability:.6f} | "
                      f"EV(P1): {expected_value / (i + 1):.4f}")
        
        return self.info_sets
    
    def solve(self, game: KuhnPoker, iterations: int = 10000, 
              verbose: bool = True) -> Dict[str, InformationSet]:
        """Solve the game using CFR."""
        return self.train(game, iterations, verbose)
    
    def get_exploitability(self, game: KuhnPoker, 
                          strategy: Dict[str, InformationSet]) -> float:
        """
        Compute exploitability of a strategy.
        
        Exploitability = (Best response value for P1) + (Best response value for P2)
        
        A Nash equilibrium has exploitability = 0.
        
        Finance Parallel:
        - Exploitability measures how much money a counter-strategy could extract
        - Market makers minimize exploitability to avoid being picked off
        """
        # Compute best response for each player
        br_value_p1 = self._best_response_value(game, strategy, Player.PLAYER_1)
        br_value_p2 = self._best_response_value(game, strategy, Player.PLAYER_2)
        
        return br_value_p1 + br_value_p2
    
    def _best_response_value(self, game: KuhnPoker, 
                             strategy: Dict[str, InformationSet],
                             br_player: Player) -> float:
        """Compute the value of the best response for one player."""
        total_value = 0.0
        
        for initial_state, prob in game.get_all_initial_states():
            value = self._br_traverse(initial_state, strategy, br_player, 1.0)
            total_value += prob * value
        
        return total_value
    
    def _br_traverse(self, state: KuhnState, strategy: Dict[str, InformationSet],
                     br_player: Player, reach_prob: float) -> float:
        """Traverse game tree computing best response value."""
        if state.is_terminal():
            payoffs = state.get_payoffs()
            return payoffs[br_player]
        
        player = state.get_current_player()
        legal_actions = state.get_legal_actions()
        info_key = state.get_information_set_key(player)
        
        if player == br_player:
            # Best response: choose action with maximum value
            best_value = float('-inf')
            for action in legal_actions:
                next_state = state.apply_action(action)
                value = self._br_traverse(next_state, strategy, br_player, reach_prob)
                best_value = max(best_value, value)
            return best_value
        else:
            # Opponent plays according to strategy
            if info_key in strategy:
                strat = strategy[info_key].get_average_strategy()
            else:
                strat = np.ones(len(legal_actions)) / len(legal_actions)
            
            expected_value = 0.0
            for i, action in enumerate(legal_actions):
                next_state = state.apply_action(action)
                value = self._br_traverse(next_state, strategy, br_player, 
                                         reach_prob * strat[i])
                expected_value += strat[i] * value
            
            return expected_value


# ==================== Analysis & Visualization ====================

def print_strategy(info_sets: Dict[str, InformationSet]):
    """
    Print the converged strategy in readable format.
    
    This is the Nash equilibrium strategy for Kuhn Poker.
    """
    print("\n" + "=" * 60)
    print("KUHN POKER NASH EQUILIBRIUM STRATEGY")
    print("=" * 60)
    
    # Known Nash equilibrium for comparison:
    # P1 with J: bet 1/3, check 2/3; facing bet after check: fold
    # P1 with Q: check; facing bet: call 1/3  
    # P1 with K: bet α, check 1-α; facing bet: call
    # P2 with J facing check: bet 1/3; facing bet: fold
    # P2 with Q facing check: check; facing bet: call 1/3
    # P2 with K facing check: bet; facing bet: call
    
    cards = ['J', 'Q', 'K']
    
    for card in cards:
        print(f"\n{'='*20} Holding {card} {'='*20}")
        
        for key in sorted(info_sets.keys()):
            if key.startswith(card):
                info_set = info_sets[key]
                history = key[1:] if len(key) > 1 else "(first to act)"
                
                strategy = info_set.get_average_strategy()
                actions = info_set.legal_actions
                
                print(f"  After '{history:10s}': ", end="")
                for i, action in enumerate(actions):
                    print(f"{action.name}={strategy[i]:.1%}  ", end="")
                print()


def get_finance_analogs() -> List[FinanceAnalog]:
    """Return game-to-finance concept mappings for Kuhn Poker."""
    return [
        FinanceAnalog(
            game_concept="Bluffing with Jack (weakest hand)",
            finance_concept="Overwriting covered calls on weak positions",
            description="Taking aggressive action with weak holdings to gain edge",
            example="Betting with J is ~33% optimal; similar to selling options "
                   "on positions you expect to be called/exercised"
        ),
        FinanceAnalog(
            game_concept="Value betting with King (strongest hand)",
            finance_concept="Aggressive pricing when holding informational edge",
            description="Extracting maximum value when you know you're ahead",
            example="Betting with K is always optimal, like widening spreads "
                   "when you have insider information advantage"
        ),
        FinanceAnalog(
            game_concept="Calling frequency at Nash equilibrium",
            finance_concept="Market maker quote adjustment",
            description="Balancing defense against exploitation",
            example="Optimal call frequency makes opponent indifferent, "
                   "like MM spread that makes arbitrage unprofitable"
        ),
        FinanceAnalog(
            game_concept="Exploitability metric",
            finance_concept="Strategy's P&L leakage to sophisticated traders",
            description="How much edge a perfect counter-strategy could extract",
            example="Lower exploitability = more robust to adverse selection"
        ),
        FinanceAnalog(
            game_concept="Regret minimization",
            finance_concept="Adaptive portfolio rebalancing",
            description="Adjusting strategy based on counterfactual analysis",
            example="'If I had hedged differently, I would have made X' "
                   "→ adjust future hedging proportionally"
        ),
    ]


def analyze_strategy_entropy(info_sets: Dict[str, InformationSet]) -> Dict[str, float]:
    """
    Analyze strategy entropy at each decision point.
    
    Higher entropy = more randomized = harder to exploit.
    Pure strategies (entropy=0) are easy to exploit.
    """
    entropies = {}
    for key, info_set in info_sets.items():
        strategy = info_set.get_average_strategy()
        entropies[key] = entropy(strategy)
    return entropies


# ==================== Main ====================

if __name__ == "__main__":
    print("=" * 60)
    print("KUHN POKER CFR SOLVER")
    print("Demonstrating Counterfactual Regret Minimization")
    print("=" * 60)
    
    # Create game and solver
    game = KuhnPoker()
    solver = KuhnCFR()
    
    # Train CFR
    print("\nTraining CFR for 10,000 iterations...\n")
    info_sets = solver.solve(game, iterations=10000)
    
    # Print results
    print_strategy(info_sets)
    
    # Print finance analogs
    print("\n" + "=" * 60)
    print("FINANCE APPLICATIONS")
    print("=" * 60)
    
    for analog in get_finance_analogs():
        print(f"\n🎲 Game: {analog.game_concept}")
        print(f"💰 Finance: {analog.finance_concept}")
        print(f"📝 {analog.description}")
    
    # Analyze entropy
    print("\n" + "=" * 60)
    print("STRATEGY ENTROPY ANALYSIS")
    print("(Higher = more mixed = harder to exploit)")
    print("=" * 60)
    
    entropies = analyze_strategy_entropy(info_sets)
    for key, ent in sorted(entropies.items(), key=lambda x: -x[1]):
        print(f"{key:30s}: {ent:.4f} bits")
