#!/usr/bin/env python3
"""
GameTheory Engine - Main Demo

This script demonstrates the complete pipeline:
1. Train CFR on Kuhn Poker
2. Analyze the Nash equilibrium strategy
3. Translate to market making parameters
4. Visualize the results

Run with: python main.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.games.poker.kuhn_cfr import (
    KuhnPoker, KuhnCFR, print_strategy, 
    get_finance_analogs, analyze_strategy_entropy
)
from src.finance.translator import (
    PokerToMarketMaking, ExploitabilityAnalysis, 
    generate_finance_report
)
from src.visualization.strategy_viewer import (
    visualize_game_tree_ascii, export_strategy_json
)


def main():
    print("=" * 70)
    print("  GAMETHEORY ENGINE - COMPREHENSIVE DEMO")
    print("  Unifying Game Theory, AI, and Quantitative Finance")
    print("=" * 70)
    print()
    
    # ==================== PHASE 1: CFR Training ====================
    print("PHASE 1: Training Counterfactual Regret Minimization")
    print("-" * 70)
    print("CFR iteratively minimizes regret to converge to Nash equilibrium.")
    print("This is the same algorithm used by superhuman poker AI systems.")
    print()
    
    game = KuhnPoker()
    solver = KuhnCFR()
    
    # Train with progress
    print("Training for 50,000 iterations...")
    info_sets = solver.solve(game, iterations=50000, verbose=True)
    
    # ==================== PHASE 2: Strategy Analysis ====================
    print("\n" + "=" * 70)
    print("PHASE 2: Nash Equilibrium Strategy Analysis")
    print("-" * 70)
    
    print_strategy(info_sets)
    
    # Show known Nash equilibrium comparison
    print("\n" + "-" * 70)
    print("KEY INSIGHTS:")
    print("-" * 70)
    print("""
The Nash equilibrium reveals optimal play:

1. BLUFFING (Jack): 
   - P1 should bluff ~33% when first to act
   - P2 should bluff ~33% when P1 checks
   → Finance: Aggression with weak positions (tight quotes)

2. VALUE BETTING (King):
   - Always bet/raise with the nuts
   → Finance: Maximize when you have edge

3. MARGINAL HANDS (Queen):
   - Never value bet, but call bets ~33%
   → Finance: Don't overplay uncertain positions

4. INDIFFERENCE:
   - Opponent is made indifferent to calling/folding
   → Finance: GTO pricing makes arbitrage impossible
    """)
    
    # ==================== PHASE 3: Finance Translation ====================
    print("\n" + "=" * 70)
    print("PHASE 3: Game Theory → Quantitative Finance Translation")
    print("-" * 70)
    
    exploitability = solver.get_exploitability(game, info_sets)
    report = generate_finance_report(info_sets, exploitability)
    print(report)
    
    # ==================== PHASE 4: Strategy Entropy ====================
    print("\n" + "=" * 70)
    print("PHASE 4: Information-Theoretic Analysis")
    print("-" * 70)
    print("""
Strategy entropy measures unpredictability (bits of uncertainty).
- High entropy = mixed strategy = hard to exploit
- Low entropy = pure strategy = predictable = exploitable

Finance parallel: Predictable order flow gets front-run.
    """)
    
    entropies = analyze_strategy_entropy(info_sets)
    
    print("\nEntropy by Decision Point:")
    print("-" * 40)
    for key, ent in sorted(entropies.items(), key=lambda x: -x[1]):
        interpretation = "MIXED (robust)" if ent > 0.5 else "PURE (exploitable)"
        print(f"  {key:10s}: {ent:.3f} bits - {interpretation}")
    
    # ==================== PHASE 5: Visualization ====================
    print("\n" + "=" * 70)
    print("PHASE 5: Game Tree Visualization")
    print("-" * 70)
    
    print(visualize_game_tree_ascii(info_sets))
    
    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("SUMMARY: Why This Matters for Quant Trading")
    print("=" * 70)
    print("""
This project demonstrates understanding of:

1. ALGORITHM KNOWLEDGE
   - CFR and regret minimization (directly used in trading)
   - Nash equilibrium computation
   - Exploitability analysis

2. MATHEMATICAL FOUNDATIONS
   - Information theory (entropy, KL divergence)
   - Game theory (incomplete information, mixed strategies)
   - Optimization (convex combination, regret matching)

3. PRACTICAL APPLICATIONS
   - Market making spread optimization
   - Adaptive strategy adjustment
   - Risk/reward analysis under uncertainty

4. SOFTWARE ENGINEERING
   - Clean abstractions and interfaces
   - Modular, extensible design
   - Production-quality code

Interview talking points:
- "CFR is how I'd approach adaptive market making"
- "Exploitability measures strategy robustness to adverse selection"
- "Nash equilibrium is the finance equivalent of no-arbitrage pricing"
    """)
    
    print("\n" + "=" * 70)
    print("Demo complete! Explore the codebase:")
    print("  - src/core/game_interface.py    (abstract interfaces)")
    print("  - src/games/poker/kuhn_cfr.py   (CFR implementation)")
    print("  - src/finance/translator.py     (finance applications)")
    print("=" * 70)


if __name__ == "__main__":
    main()
