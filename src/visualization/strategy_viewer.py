"""
Strategy Visualization Module

Provides text-based and data export visualizations for game-theoretic strategies.
Can be extended with matplotlib/plotly for graphical output.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.game_interface import InformationSet, entropy


def strategy_to_ascii_bar(strategy: np.ndarray, actions: List[str], width: int = 40) -> str:
    """Create ASCII bar chart for a strategy distribution."""
    lines = []
    for i, (prob, action) in enumerate(zip(strategy, actions)):
        bar_length = int(prob * width)
        bar = "█" * bar_length + "░" * (width - bar_length)
        lines.append(f"  {str(action):8s} [{bar}] {prob:5.1%}")
    return "\n".join(lines)


def visualize_game_tree_ascii(info_sets: Dict[str, InformationSet], 
                              max_depth: int = 3) -> str:
    """
    Create ASCII visualization of game tree with strategies.
    
    Shows decision points and strategy distributions.
    """
    lines = []
    lines.append("GAME TREE VISUALIZATION")
    lines.append("=" * 60)
    lines.append("")
    
    # Group by card
    cards = ['J', 'Q', 'K']
    card_names = {'J': 'Jack (Weak)', 'Q': 'Queen (Medium)', 'K': 'King (Strong)'}
    
    for card in cards:
        lines.append(f"┌─ {card_names[card]} ─────────────────────────────────────────┐")
        
        # Get all info sets for this card, sorted by history length
        card_sets = [(k, v) for k, v in info_sets.items() if k.startswith(card)]
        card_sets.sort(key=lambda x: len(x[0]))
        
        for key, info_set in card_sets:
            history = key[1:] if len(key) > 1 else "START"
            strategy = info_set.get_average_strategy()
            actions = [str(a).split('.')[-1] for a in info_set.legal_actions]
            
            indent = "│  " + "   " * len(key[1:])
            
            lines.append(f"│")
            lines.append(f"│  [{history}] Decision Point")
            
            for prob, action in zip(strategy, actions):
                bar = "█" * int(prob * 20)
                lines.append(f"│     → {action:6s}: {bar:20s} {prob:5.1%}")
        
        lines.append(f"└{'─' * 58}┘")
        lines.append("")
    
    return "\n".join(lines)


def export_strategy_json(info_sets: Dict[str, InformationSet]) -> str:
    """Export strategy as JSON for external visualization tools."""
    data = {
        "metadata": {
            "game": "Kuhn Poker",
            "algorithm": "CFR",
        },
        "strategies": {}
    }
    
    for key, info_set in info_sets.items():
        strategy = info_set.get_average_strategy().tolist()
        actions = [str(a).split('.')[-1] for a in info_set.legal_actions]
        
        data["strategies"][key] = {
            "actions": actions,
            "probabilities": strategy,
            "entropy": entropy(info_set.get_average_strategy()),
        }
    
    return json.dumps(data, indent=2)


def convergence_report(exploitability_history: List[float]) -> str:
    """Generate convergence analysis report."""
    lines = []
    lines.append("CONVERGENCE ANALYSIS")
    lines.append("=" * 60)
    
    if len(exploitability_history) < 2:
        lines.append("Insufficient data for convergence analysis")
        return "\n".join(lines)
    
    final = exploitability_history[-1]
    initial = exploitability_history[0]
    improvement = (initial - final) / initial * 100
    
    lines.append(f"Initial exploitability: {initial:.6f}")
    lines.append(f"Final exploitability:   {final:.6f}")
    lines.append(f"Improvement:            {improvement:.1f}%")
    lines.append("")
    
    # ASCII convergence plot
    lines.append("Convergence Plot (exploitability over iterations):")
    lines.append("")
    
    max_exp = max(exploitability_history)
    height = 10
    width = min(len(exploitability_history), 50)
    step = len(exploitability_history) // width
    
    sampled = exploitability_history[::step][:width]
    
    for row in range(height, 0, -1):
        threshold = max_exp * row / height
        line = "│"
        for val in sampled:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        lines.append(line)
    
    lines.append("└" + "─" * width)
    lines.append(f" 0{' ' * (width - 10)}iterations")
    
    return "\n".join(lines)


def strategy_comparison(strategy1: Dict[str, InformationSet],
                       strategy2: Dict[str, InformationSet],
                       name1: str = "Strategy 1",
                       name2: str = "Strategy 2") -> str:
    """Compare two strategies side by side."""
    lines = []
    lines.append("STRATEGY COMPARISON")
    lines.append("=" * 70)
    lines.append(f"Comparing: {name1} vs {name2}")
    lines.append("")
    
    all_keys = set(strategy1.keys()) | set(strategy2.keys())
    
    for key in sorted(all_keys):
        lines.append(f"\n[{key}]")
        
        if key in strategy1:
            s1 = strategy1[key].get_average_strategy()
            actions1 = [str(a).split('.')[-1] for a in strategy1[key].legal_actions]
        else:
            s1, actions1 = None, None
            
        if key in strategy2:
            s2 = strategy2[key].get_average_strategy()
            actions2 = [str(a).split('.')[-1] for a in strategy2[key].legal_actions]
        else:
            s2, actions2 = None, None
        
        if s1 is not None and s2 is not None:
            for i, action in enumerate(actions1):
                diff = s2[i] - s1[i]
                diff_str = f"+{diff:.1%}" if diff > 0 else f"{diff:.1%}"
                lines.append(f"  {action}: {s1[i]:.1%} → {s2[i]:.1%} ({diff_str})")
        elif s1 is not None:
            lines.append(f"  {name1} only: {dict(zip(actions1, s1))}")
        else:
            lines.append(f"  {name2} only: {dict(zip(actions2, s2))}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    from src.games.poker.kuhn_cfr import KuhnPoker, KuhnCFR
    
    print("Training CFR and generating visualizations...")
    print()
    
    game = KuhnPoker()
    solver = KuhnCFR()
    info_sets = solver.solve(game, iterations=20000, verbose=False)
    
    # ASCII game tree
    print(visualize_game_tree_ascii(info_sets))
    
    # JSON export
    print("\n" + "=" * 60)
    print("JSON EXPORT (for D3.js / React visualization):")
    print("=" * 60)
    print(export_strategy_json(info_sets)[:500] + "...")
