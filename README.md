# Hanafuda-AI

> **This is a fork of [guansanghai/KoiKoi-AI](https://github.com/guansanghai/KoiKoi-AI)**, refactored with improved code structure and design patterns.

### Original Paper
>S. Guan, J. Wang, R. Zhu, J. Qian and Z. Wei, **"Learning to Play Koi-Koi Hanafuda Card Games with Transformers,"** *IEEE Transactions on Artificial Intelligence*, vol. 4, no. 6, pp. 1449-1460, 2023. [doi: 10.1109/TAI.2023.3240674](https://ieeexplore.ieee.org/document/10032777).

Learning based AI for playing multi-round Koi-Koi hanafuda card games. ([@guansanghai](https://github.com/guansanghai))

![Play Interface](/markdown/Kapture.gif)

## Installation

```bash
# Clone the repository
git clone https://github.com/ttooleey/Hanafuda-AI.git
cd Hanafuda-AI

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

```bash
# Play against AI
uv run python -m koikoi.play --ai RL-Point --name YourName
```

## Package Structure

```
koikoi/
├── core/           # Core game logic
│   ├── card.py         # Card representation and encoding
│   ├── constants.py    # Game constants and enums
│   ├── game_state.py   # Multi-round game management
│   ├── round_state.py  # Single round state machine
│   └── yaku.py         # Yaku (winning hand) definitions
├── ai/             # AI agents and strategies
│   ├── agent.py        # KoiKoiAgent facade class
│   ├── models.py       # Transformer neural network models
│   └── strategies.py   # Action selection strategies
├── training/       # Training utilities
│   ├── buffer.py       # Experience replay buffer
│   ├── simulator.py    # Self-play simulation
│   └── trainer.py      # Training loop management
├── ui/             # User interface
│   └── gui.py          # FreeSimpleGUI-based GUI
└── utils/          # Utility functions
    └── helpers.py      # Common helper functions
```

## Environment

* Python 3.9
* PyTorch 1.8.1
* FreeSimpleGUI (PySimpleGUI compatible fork)

## About Koi-Koi Hanafuda Card Games

[Hanafuda](https://en.wikipedia.org/wiki/Hanafuda) is a kind of traditional Japanese playing cards. A hanafuda deck contains 48 cards divided by 12 suits corresponding to 12 months, which are also divided into four rank-like categories with different importance. [Koi-Koi](https://en.wikipedia.org/wiki/Koi-Koi) is a kind of two-player hanafuda card game. The goal of Koi-Koi is to collect cards by matching the cards by suit, and forming specific winning hands called Yaku from the acquired pile to earn points from the opponent.

![Hanafuda Deck](/markdown/koikoi_deck.png)

## Rules & Yaku List

Koi-Koi is consisted by multiple rounds and both players start with equal points. In every round, two players discard and draw to pair and collect cards by turn until someone forms Yakus successfully. Then, he can end this round to receive points from the opponent, or claim koi-koi and continues this round to earn more yakus and points. The detailed rules and Yaku list of this project is the same as PC game [KoiKoi-Japan](https://store.steampowered.com/app/364930/KoiKoi_Japan_Hanafuda_playing_cards/) on Steam.

![Yaku List](/markdown/koikoi_yaku.png)

## Architecture

### Transformer-based Neural Network

The AI uses a Transformer encoder architecture with:
- 2 encoder layers with 4 attention heads
- 48-dimensional embeddings (one per card)
- Multi-head attention for learning card relationships

### Training Methods

1. **Supervised Learning (SL)**: Pre-training on expert game records
2. **Reinforcement Learning (RL)**: Monte-Carlo RL with self-play
   - RL-Point: Optimized for points per round
   - RL-WP: Optimized for win probability

### Design Patterns

The codebase follows clean code principles with:
- **Strategy Pattern**: Interchangeable action selection algorithms
- **State Pattern**: Game phase management
- **Facade Pattern**: Simplified agent interface

## API Usage

```python
from koikoi import KoiKoiGameState, KoiKoiAgent

# Create a new game
game = KoiKoiGameState()

# Load a pretrained agent
agent = KoiKoiAgent.load_pretrained("model_agent/")

# Play a game
while not game.game_over:
    action = agent.select_action(game, game.get_action_mask())
    game.round_state.step(action)
```

## License

MIT License
