"""
Finance Translation Layer for GameTheory Engine

This module maps game-theoretic concepts to quantitative finance applications,
demonstrating the deep connections between optimal game play and trading strategy.

Key Mappings:
- CFR Regret Minimization → Adaptive Market Making
- Nash Equilibrium → GTO Pricing
- Bluffing Frequency → Information Asymmetry Exploitation
- Exploitability → Strategy Robustness to Adverse Selection
- Information Sets → Incomplete Information (Options, Dark Pools)

This is the "killer feature" that shows understanding of why game theory
matters for quantitative trading.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.game_interface import InformationSet, FinanceAnalog, entropy


# ==================== Core Financial Concepts ====================

@dataclass
class MarketMakerState:
    """
    Represents a market maker's decision state.
    
    Analogous to a poker player's information set:
    - inventory = hand strength
    - order_flow_imbalance = betting pattern signals
    - volatility = pot odds
    """
    inventory: float  # Current position (-1 to 1, normalized)
    order_flow_imbalance: float  # Recent buy vs sell pressure
    volatility: float  # Current market volatility
    spread: float  # Current bid-ask spread
    
    def to_risk_score(self) -> float:
        """Compute risk score analogous to hand strength."""
        # Higher inventory + higher volatility = more risk
        return abs(self.inventory) * (1 + self.volatility)


@dataclass  
class OptionPosition:
    """
    Options position for hedging analysis.
    
    Maps to poker concepts:
    - Delta = probability of winning (like pot equity)
    - Gamma = sensitivity to new information
    - Theta = time decay (like blinds eroding stack)
    - Vega = sensitivity to volatility
    """
    delta: float
    gamma: float
    theta: float
    vega: float
    
    def get_hedge_ratio(self) -> float:
        """Optimal hedge ratio like pot odds."""
        return -self.delta


# ==================== Poker to Market Making Translation ====================

class PokerToMarketMaking:
    """
    Translates poker CFR strategies to market making decisions.
    
    Core insight: Both poker and market making involve:
    1. Managing inventory/position under uncertainty
    2. Balancing aggression vs defense  
    3. Exploiting information asymmetry
    4. Adapting to opponent behavior
    """
    
    @staticmethod
    def bluff_frequency_to_quote_aggression(bluff_freq: float) -> float:
        """
        Map poker bluffing frequency to quote aggression.
        
        Higher bluff frequency = tighter spreads (more aggressive)
        Lower bluff frequency = wider spreads (more defensive)
        
        In poker: bluffing wins pots without showdown
        In MM: tight spreads win order flow without information edge
        """
        # Normalize to spread multiplier (0.5 to 2.0)
        return 2.0 - bluff_freq * 1.5
    
    @staticmethod
    def call_frequency_to_inventory_tolerance(call_freq: float) -> float:
        """
        Map poker calling frequency to inventory tolerance.
        
        Higher call frequency = willing to hold more inventory
        Lower call frequency = faster inventory unwinding
        
        In poker: calling = willing to see showdown
        In MM: holding inventory = willing to face adverse selection
        """
        return call_freq
    
    @staticmethod
    def translate_strategy(poker_strategy: Dict[str, InformationSet]) -> Dict[str, Any]:
        """
        Convert a complete Kuhn Poker strategy to MM parameters.
        
        Returns actionable market making parameters based on game theory.
        """
        # Extract key frequencies
        # Jack (weak hand) bluff frequency → aggression when weak
        j_bluff = poker_strategy.get('J', None)
        j_bluff_freq = j_bluff.get_average_strategy()[1] if j_bluff else 0.33
        
        # Queen (medium) call frequency → tolerance when uncertain
        q_call = poker_strategy.get('Qb', None)
        q_call_freq = q_call.get_average_strategy()[1] if q_call else 0.33
        
        # King (strong) value bet → aggression when strong
        k_bet = poker_strategy.get('K', None)
        k_bet_freq = k_bet.get_average_strategy()[1] if k_bet else 0.67
        
        return {
            'spread_multiplier': PokerToMarketMaking.bluff_frequency_to_quote_aggression(j_bluff_freq),
            'inventory_tolerance': PokerToMarketMaking.call_frequency_to_inventory_tolerance(q_call_freq),
            'value_aggression': k_bet_freq,
            'interpretation': {
                'spread_multiplier': f"Base spread x {PokerToMarketMaking.bluff_frequency_to_quote_aggression(j_bluff_freq):.2f}",
                'inventory_tolerance': f"Max inventory = {q_call_freq:.1%} of capital",
                'value_aggression': f"Quote inside {k_bet_freq:.1%} of time when edge detected"
            },
            'finance_insights': [
                f"When position is weak (like J in poker), maintain {j_bluff_freq:.1%} aggressive quotes",
                f"When uncertain (like Q), tolerate inventory {q_call_freq:.1%} of the time",
                f"When confident (like K), push volume {k_bet_freq:.1%} of opportunities"
            ]
        }


# ==================== Regret to Portfolio Rebalancing ====================

class RegretToRebalancing:
    """
    Maps CFR regret minimization to portfolio rebalancing.
    
    Core insight: Both involve learning from counterfactual outcomes
    and adjusting allocations proportionally.
    """
    
    @staticmethod
    def regrets_to_allocation_adjustments(regret_sum: np.ndarray, 
                                           current_allocation: np.ndarray) -> np.ndarray:
        """
        Convert CFR regrets to portfolio adjustment signals.
        
        Positive regret = "I should have allocated more here"
        Negative regret = "I allocated too much here"
        
        Returns suggested allocation adjustment.
        """
        positive_regrets = np.maximum(regret_sum, 0)
        total = np.sum(positive_regrets)
        
        if total > 0:
            target_allocation = positive_regrets / total
        else:
            target_allocation = np.ones(len(regret_sum)) / len(regret_sum)
        
        return target_allocation - current_allocation
    
    @staticmethod
    def counterfactual_pnl(actual_allocation: np.ndarray,
                          asset_returns: np.ndarray,
                          all_possible_allocations: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute counterfactual P&L analysis.
        
        "What if I had allocated differently?"
        This is the finance analog of CFR's regret computation.
        """
        actual_return = np.dot(actual_allocation, asset_returns)
        
        results = {
            'actual_return': actual_return,
            'counterfactual_returns': {},
            'regrets': {}
        }
        
        for i, alt_allocation in enumerate(all_possible_allocations):
            alt_return = np.dot(alt_allocation, asset_returns)
            results['counterfactual_returns'][f'allocation_{i}'] = alt_return
            results['regrets'][f'allocation_{i}'] = alt_return - actual_return
        
        return results


# ==================== Exploitability to Strategy Robustness ====================

class ExploitabilityAnalysis:
    """
    Maps game exploitability to trading strategy robustness.
    
    Core insight: Just as exploitable poker strategies lose to adaptive opponents,
    predictable trading strategies lose to sophisticated traders.
    """
    
    @staticmethod
    def strategy_entropy_analysis(strategy: np.ndarray) -> Dict[str, float]:
        """
        Analyze strategy entropy as a measure of unpredictability.
        
        Higher entropy = harder to exploit = more robust
        Lower entropy = predictable = vulnerable to front-running
        """
        strat_entropy = entropy(strategy)
        max_entropy = np.log2(len(strategy))
        
        return {
            'entropy_bits': strat_entropy,
            'max_entropy_bits': max_entropy,
            'entropy_ratio': strat_entropy / max_entropy if max_entropy > 0 else 0,
            'predictability_score': 1 - (strat_entropy / max_entropy if max_entropy > 0 else 0),
            'interpretation': 'High' if strat_entropy > 0.5 * max_entropy else 'Low',
            'finance_implication': (
                'Strategy is well-randomized, hard to front-run' 
                if strat_entropy > 0.5 * max_entropy 
                else 'Strategy is predictable, vulnerable to adverse selection'
            )
        }
    
    @staticmethod
    def compute_strategy_robustness(exploitability: float, 
                                    baseline_exploitability: float = 0.055) -> Dict[str, Any]:
        """
        Convert exploitability to robustness score.
        
        Args:
            exploitability: Strategy's exploitability (lower = better)
            baseline_exploitability: Theoretical minimum (Nash equilibrium)
        
        Returns:
            Robustness metrics and trading implications
        """
        excess_exploitability = max(0, exploitability - baseline_exploitability)
        robustness_score = 1 / (1 + excess_exploitability * 10)
        
        return {
            'exploitability': exploitability,
            'baseline': baseline_exploitability,
            'excess': excess_exploitability,
            'robustness_score': robustness_score,
            'grade': (
                'A' if robustness_score > 0.95 else
                'B' if robustness_score > 0.85 else
                'C' if robustness_score > 0.70 else
                'D' if robustness_score > 0.50 else 'F'
            ),
            'pnl_leakage_estimate': f"{excess_exploitability * 100:.2f}% per unit of volume",
            'recommendation': (
                'Strategy is near-optimal, minimal improvements possible'
                if robustness_score > 0.95 else
                'Consider reducing predictable patterns'
                if robustness_score > 0.70 else
                'Strategy has significant leakage to sophisticated traders'
            )
        }


# ==================== Complete Translation Report ====================

def generate_finance_report(poker_strategy: Dict[str, InformationSet],
                           exploitability: float) -> str:
    """
    Generate a complete finance translation report from poker strategy.
    
    This is the "money slide" that shows understanding of game theory
    applications to quantitative finance.
    """
    mm_params = PokerToMarketMaking.translate_strategy(poker_strategy)
    robustness = ExploitabilityAnalysis.compute_strategy_robustness(exploitability)
    
    # Analyze strategy entropy
    entropy_analysis = {}
    for key, info_set in poker_strategy.items():
        strategy = info_set.get_average_strategy()
        entropy_analysis[key] = ExploitabilityAnalysis.strategy_entropy_analysis(strategy)
    
    report = []
    report.append("=" * 70)
    report.append("GAME THEORY → QUANTITATIVE FINANCE TRANSLATION REPORT")
    report.append("=" * 70)
    
    report.append("\n## 1. MARKET MAKING PARAMETERS (from Kuhn Poker Nash Equilibrium)")
    report.append("-" * 70)
    for key, value in mm_params['interpretation'].items():
        report.append(f"  {key}: {value}")
    
    report.append("\n## 2. TRADING INSIGHTS")
    report.append("-" * 70)
    for insight in mm_params['finance_insights']:
        report.append(f"  • {insight}")
    
    report.append("\n## 3. STRATEGY ROBUSTNESS ANALYSIS")
    report.append("-" * 70)
    report.append(f"  Exploitability: {robustness['exploitability']:.4f}")
    report.append(f"  Robustness Score: {robustness['robustness_score']:.2%}")
    report.append(f"  Grade: {robustness['grade']}")
    report.append(f"  Est. P&L Leakage: {robustness['pnl_leakage_estimate']}")
    report.append(f"  Recommendation: {robustness['recommendation']}")
    
    report.append("\n## 4. DECISION POINT ENTROPY (Predictability Analysis)")
    report.append("-" * 70)
    for key, analysis in sorted(entropy_analysis.items()):
        report.append(f"  {key:10s}: Entropy={analysis['entropy_bits']:.3f} bits | "
                     f"Predictability={analysis['predictability_score']:.1%} | "
                     f"{analysis['interpretation']}")
    
    report.append("\n## 5. KEY ANALOGIES")
    report.append("-" * 70)
    
    analogies = [
        ("Poker Bluffing", "Market Making Spread Tightening",
         "Both are aggressive actions taken with weak positions to win without 'showdown'"),
        ("Pot Odds", "Risk/Reward Ratio",
         "Both determine whether to continue with uncertain outcomes"),
        ("Hand Reading", "Order Flow Analysis", 
         "Both involve inferring hidden information from observable actions"),
        ("Nash Equilibrium", "No-Arbitrage Pricing",
         "Both represent states where no player can unilaterally improve"),
        ("Regret Minimization", "Adaptive Hedging",
         "Both adjust strategy based on 'what could have been'"),
        ("Exploitability", "Strategy Alpha Decay",
         "Both measure how much edge a counter-strategy could extract"),
    ]
    
    for game, finance, desc in analogies:
        report.append(f"\n  🎲 {game}")
        report.append(f"  💰 {finance}")
        report.append(f"     {desc}")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return "\n".join(report)


# ==================== Demo Application ====================

if __name__ == "__main__":
    # Import Kuhn Poker solver
    from src.games.poker.kuhn_cfr import KuhnPoker, KuhnCFR
    
    print("Training Kuhn Poker CFR to generate finance translation...")
    print()
    
    # Solve Kuhn Poker
    game = KuhnPoker()
    solver = KuhnCFR()
    strategy = solver.solve(game, iterations=50000, verbose=False)
    
    # Get exploitability
    exploitability = solver.get_exploitability(game, strategy)
    
    # Generate finance report
    report = generate_finance_report(strategy, exploitability)
    print(report)
    
    # Interactive demo
    print("\n" + "=" * 70)
    print("INTERACTIVE DEMO: Portfolio Rebalancing via Regret Minimization")
    print("=" * 70)
    
    # Simulate portfolio scenario
    current_allocation = np.array([0.4, 0.3, 0.3])  # Stocks, Bonds, Cash
    asset_returns = np.array([0.05, 0.02, 0.01])     # This period's returns
    
    # Alternative allocations we could have taken
    alternatives = [
        np.array([0.6, 0.2, 0.2]),  # More aggressive
        np.array([0.2, 0.5, 0.3]),  # More conservative
        np.array([0.33, 0.33, 0.34]),  # Equal weight
    ]
    
    cf_analysis = RegretToRebalancing.counterfactual_pnl(
        current_allocation, asset_returns, alternatives
    )
    
    print(f"\nCurrent allocation: {current_allocation}")
    print(f"Actual return: {cf_analysis['actual_return']:.2%}")
    print("\nCounterfactual analysis (regret-style):")
    for key, regret in cf_analysis['regrets'].items():
        print(f"  {key}: regret = {regret:+.2%}")
    
    # Compute adjustment based on regrets
    regret_vector = np.array([cf_analysis['regrets'][f'allocation_{i}'] 
                             for i in range(len(alternatives))])
    
    print(f"\nCFR-style interpretation:")
    print("  Actions with positive regret → increase allocation")
    print("  Actions with negative regret → decrease allocation")
