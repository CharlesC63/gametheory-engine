# GameTheory Engine

A unified decision-making framework implementing game-theoretic algorithms across Chess, Shogi, Poker, and Go — with direct applications to quantitative finance.

## 🎯 Core Philosophy

The key insight: **optimal decision-making under uncertainty follows universal principles** that translate across games and financial markets.

| Game | Algorithm | Finance Application |
|------|-----------|---------------------|
| Poker | Counterfactual Regret Minimization (CFR) | Market making, options hedging |
| Chess/Shogi | Minimax + Alpha-Beta | Adversarial optimization, worst-case analysis |
| Go | Monte Carlo Tree Search (MCTS) | Portfolio simulation, scenario analysis |
| All | Nash Equilibrium | Game-theoretic optimal (GTO) pricing |

## 🏗️ Architecture

```
src/
├── core/           # Abstract game interfaces & shared algorithms
├── games/
│   ├── poker/      # CFR implementation (Kuhn, Texas Hold'em)
│   ├── chess/      # Minimax with alpha-beta pruning
│   ├── go/         # MCTS implementation
│   └── shogi/      # Extended minimax for drops
├── finance/        # Game-to-finance translation layer
└── visualization/  # Interactive game tree & strategy visualizations
```

## 🚀 Quick Start (VSCode)

### 1. Open in VSCode
```bash
# Unzip and open the folder in VSCode
code gametheory-engine
```

### 2. Create Virtual Environment
```bash
# In VSCode terminal (Ctrl+`)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -e ".[dev]"
```

### 4. Run the Demo
```bash
python main.py
```

### 5. Run Tests
```bash
pytest tests/ -v
```

## 🎮 VSCode Features

This project includes VSCode configuration for:
- **Debugging**: Press F5 to run with debugger (multiple launch configs)
- **Testing**: Use the Testing sidebar to run/debug tests
- **Formatting**: Auto-format on save with Black
- **IntelliSense**: Full type hints for autocomplete

## 📊 Key Features

### 1. Counterfactual Regret Minimization (CFR)
- Vanilla CFR, CFR+, and Monte Carlo CFR variants
- Exploitability calculation
- Strategy convergence visualization

### 2. Game-to-Finance Translation
- Poker ranges → Options delta hedging
- Bluffing frequency → Information asymmetry exploitation
- Position sizing → Kelly Criterion optimization

### 3. Cross-Game Insights
- Unified "exploitability" metric across all games
- Information-theoretic complexity analysis
- Transferable strategic patterns

## 📈 Quantitative Finance Applications

This framework demonstrates understanding of:
- **Regret minimization** — core to adaptive trading strategies
- **Nash equilibria** — optimal pricing in competitive markets
- **Information sets** — modeling incomplete information (options, dark pools)
- **Exploitability** — measuring strategy robustness

## 🎓 Educational Value

Perfect for understanding:
- Why GTO poker strategy parallels market making
- How MCTS simulation relates to Monte Carlo option pricing
- The mathematics of decision-making under uncertainty

## Author

Charles — Graduate Student, Villanova University  
MS Computer Science (May 2025) | MS Finance (May 2026)

## License

MIT License
