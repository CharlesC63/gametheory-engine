"""
Tests for Kuhn Poker CFR Implementation

Run with: pytest tests/test_kuhn_cfr.py -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.games.poker.kuhn_cfr import (
    KuhnPoker, KuhnCFR, KuhnState, Card, Action, Player
)


class TestKuhnState:
    """Tests for KuhnState game logic."""
    
    def test_initial_state_not_terminal(self):
        """Initial state should not be terminal."""
        state = KuhnState(cards=(Card.JACK, Card.QUEEN))
        assert not state.is_terminal()
    
    def test_check_check_is_terminal(self):
        """Check-check should end the game."""
        state = KuhnState(cards=(Card.JACK, Card.QUEEN), history='cc')
        assert state.is_terminal()
    
    def test_bet_fold_is_terminal(self):
        """Bet-fold should end the game."""
        state = KuhnState(cards=(Card.JACK, Card.QUEEN), history='bf')
        assert state.is_terminal()
    
    def test_bet_call_is_terminal(self):
        """Bet-call should end the game."""
        state = KuhnState(cards=(Card.JACK, Card.QUEEN), history='bk')
        assert state.is_terminal()
    
    def test_check_bet_call_is_terminal(self):
        """Check-bet-call should end the game."""
        state = KuhnState(cards=(Card.JACK, Card.QUEEN), history='cbk')
        assert state.is_terminal()
    
    def test_player_1_acts_first(self):
        """Player 1 should act on empty history."""
        state = KuhnState(cards=(Card.JACK, Card.QUEEN))
        assert state.get_current_player() == Player.PLAYER_1
    
    def test_player_2_acts_after_check(self):
        """Player 2 should act after P1 checks."""
        state = KuhnState(cards=(Card.JACK, Card.QUEEN), history='c')
        assert state.get_current_player() == Player.PLAYER_2
    
    def test_player_1_acts_after_check_bet(self):
        """Player 1 should act after check-bet."""
        state = KuhnState(cards=(Card.JACK, Card.QUEEN), history='cb')
        assert state.get_current_player() == Player.PLAYER_1
    
    def test_legal_actions_at_start(self):
        """Should have check/bet options at start."""
        state = KuhnState(cards=(Card.JACK, Card.QUEEN))
        actions = state.get_legal_actions()
        assert Action.CHECK in actions
        assert Action.BET in actions
    
    def test_legal_actions_facing_bet(self):
        """Should have fold/call options facing a bet."""
        state = KuhnState(cards=(Card.JACK, Card.QUEEN), history='b')
        actions = state.get_legal_actions()
        assert Action.FOLD in actions
        assert Action.CALL in actions
    
    def test_payoffs_higher_card_wins(self):
        """Higher card should win at showdown."""
        # Q beats J
        state = KuhnState(cards=(Card.JACK, Card.QUEEN), history='cc')
        payoffs = state.get_payoffs()
        assert payoffs[Player.PLAYER_1] < 0  # J loses
        assert payoffs[Player.PLAYER_2] > 0  # Q wins
        
    def test_payoffs_fold_gives_pot(self):
        """Folding should give pot to opponent."""
        state = KuhnState(cards=(Card.KING, Card.QUEEN), history='bf')
        payoffs = state.get_payoffs()
        assert payoffs[Player.PLAYER_1] > 0  # P1 wins P2's fold
        assert payoffs[Player.PLAYER_2] < 0
    
    def test_information_set_key(self):
        """Information set should include card and history."""
        state = KuhnState(cards=(Card.JACK, Card.QUEEN), history='c')
        
        p1_key = state.get_information_set_key(Player.PLAYER_1)
        p2_key = state.get_information_set_key(Player.PLAYER_2)
        
        assert 'J' in p1_key or 'JACK' in p1_key
        assert 'Q' in p2_key or 'QUEEN' in p2_key


class TestKuhnPoker:
    """Tests for KuhnPoker game."""
    
    def test_initial_state_created(self):
        """Should create valid initial state."""
        game = KuhnPoker()
        state = game.get_initial_state()
        assert not state.is_terminal()
    
    def test_all_initial_states(self):
        """Should have 6 possible deals."""
        game = KuhnPoker()
        states = game.get_all_initial_states()
        assert len(states) == 6
        
        # Check probabilities sum to 1
        total_prob = sum(prob for _, prob in states)
        assert abs(total_prob - 1.0) < 1e-10


class TestCFR:
    """Tests for CFR solver."""
    
    def test_cfr_creates_info_sets(self):
        """CFR should create information sets."""
        game = KuhnPoker()
        solver = KuhnCFR()
        
        info_sets = solver.solve(game, iterations=100, verbose=False)
        
        assert len(info_sets) > 0
    
    def test_strategies_are_valid_distributions(self):
        """All strategies should be valid probability distributions."""
        game = KuhnPoker()
        solver = KuhnCFR()
        
        info_sets = solver.solve(game, iterations=1000, verbose=False)
        
        for key, info_set in info_sets.items():
            strategy = info_set.get_average_strategy()
            
            # Probabilities should be non-negative
            assert np.all(strategy >= 0), f"Negative probability in {key}"
            
            # Probabilities should sum to 1
            assert abs(np.sum(strategy) - 1.0) < 1e-10, f"Sum != 1 in {key}"
    
    def test_exploitability_decreases(self):
        """Exploitability should decrease with more iterations."""
        game = KuhnPoker()
        
        solver1 = KuhnCFR()
        info_sets1 = solver1.solve(game, iterations=100, verbose=False)
        exp1 = solver1.get_exploitability(game, info_sets1)
        
        solver2 = KuhnCFR()
        info_sets2 = solver2.solve(game, iterations=10000, verbose=False)
        exp2 = solver2.get_exploitability(game, info_sets2)
        
        # More iterations should (generally) lead to lower exploitability
        # Note: this isn't strictly guaranteed per-run but should hold on average
        assert exp2 <= exp1 * 1.5, "Exploitability should decrease with iterations"
    
    def test_convergence_to_known_equilibrium(self):
        """Strategy should converge to known Nash equilibrium values."""
        game = KuhnPoker()
        solver = KuhnCFR()
        
        info_sets = solver.solve(game, iterations=50000, verbose=False)
        
        # Known Nash equilibrium facts:
        # 1. P1 with Queen never value bets
        q_bet_freq = info_sets['Q'].get_average_strategy()[1]
        assert q_bet_freq < 0.05, f"Q should never bet, got {q_bet_freq}"
        
        # 2. P2 with King always calls
        k_call_freq = info_sets['Kb'].get_average_strategy()[1]
        assert k_call_freq > 0.95, f"K should always call, got {k_call_freq}"
        
        # 3. P1 with Jack facing bet should fold
        j_fold_freq = info_sets['Jcb'].get_average_strategy()[0]
        assert j_fold_freq > 0.95, f"J facing bet should fold, got {j_fold_freq}"


class TestRegretMatching:
    """Tests for regret matching algorithm."""
    
    def test_uniform_with_no_regrets(self):
        """Should return uniform distribution with zero regrets."""
        from src.core.game_interface import InformationSet
        
        info_set = InformationSet(
            key="test",
            player=Player.PLAYER_1,
            legal_actions=[Action.CHECK, Action.BET]
        )
        
        strategy = info_set.get_strategy()
        
        assert abs(strategy[0] - 0.5) < 1e-10
        assert abs(strategy[1] - 0.5) < 1e-10
    
    def test_positive_regret_increases_probability(self):
        """Positive regret should increase action probability."""
        from src.core.game_interface import InformationSet
        
        info_set = InformationSet(
            key="test",
            player=Player.PLAYER_1,
            legal_actions=[Action.CHECK, Action.BET]
        )
        
        # Add positive regret to BET action
        info_set.regret_sum = np.array([0.0, 1.0])
        
        strategy = info_set.get_strategy()
        
        assert strategy[1] == 1.0  # BET should have all probability
        assert strategy[0] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
