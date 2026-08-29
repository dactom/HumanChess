# HumanChess

HumanChess is a neural-network chess engine designed to imitate the move choices of real human players rather than play the strongest possible chess.

Current target strength: **800-900 Elo human behaviour**

The model is trained using real Lichess games from players in this rating range.

## Quick Start

Go to the project:

```bash
cd ~/HumanChess
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

### Run HumanChess

The easiest way:

```bash
./humanchess
```

Or directly through Python:

```bash
python3 -m src.uci_engine
```

Basic UCI test:

```text
uci
isready
position startpos moves e2e4
go
```

Exit with:

```text
quit
```

## Using HumanChess in Lucas Chess

HumanChess works as a UCI chess engine.

In Lucas Chess, add a new external engine and point it to:

```text
/home/dacian/HumanChess/humanchess
```

The `humanchess` launcher starts the Python UCI engine from the correct project environment.

HumanChess has successfully played full games and tournaments in Lucas Chess.

## Current Model

The current main model was trained using:

```text
5,000 Lichess Rapid games
800-900 Elo
235,358 total positions
```

Train/test split:

```text
188,520 training positions
46,838 test positions
```

Best result:

```text
Epoch 5
Test accuracy: 25.80%
```

The older 500-game model reached approximately:

```text
17.9%
```

Current checkpoint:

```text
checkpoints/human_chess_policy_5000games.pt
```

Previous model:

```text
checkpoints/human_chess_policy_500games.pt
```

## Current Move Selection

The neural network produces scores for legal moves.

HumanChess then applies small tactical penalties and temperature-controlled sampling.

Current settings in `src/uci_engine.py`:

```python
QUEEN_BLUNDER_PENALTY = 4.0
ROOK_BLUNDER_PENALTY = 1.5
TEMPERATURE = 0.5
```

These are deliberately soft penalties.

HumanChess is supposed to make mistakes because real 800-900 Elo players make mistakes.

The tactical layer is intended to reduce extreme mistakes without turning HumanChess into a conventional strong chess engine.

## Project Structure

```text
HumanChess/
├── src/
│   ├── baselines/
│   │   ├── random_policy.py
│   │   └── move_frequency_policy.py
│   ├── board_encoding.py
│   ├── dataset.py
│   ├── model.py
│   ├── move_encoding.py
│   ├── play.py
│   ├── train.py
│   └── uci_engine.py
│
├── scripts/
│   ├── data/
│   │   ├── extract_positions.py
│   │   ├── filter_5000_games.sh
│   │   └── filter_lichess.py
│   ├── inspect/
│   │   ├── inspect_dataset.py
│   │   ├── inspect_positions.py
│   │   └── inspect_training_examples.py
│   └── benchmarks/
│       └── evaluate_baseline.py
│
├── benchmarks/
├── data/
├── checkpoints/
├── humanchess
├── requirements.txt
├── requirements-intel.txt
├── requirements-nvidia.txt
└── README.md
```

The `data/`, `checkpoints/`, `models/`, and `logs/` directories are excluded from Git because they may contain large generated files.

## Python Environment

Create the virtual environment if required:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

General dependencies are recorded in:

```text
requirements.txt
```

Intel Arc/XPU dependencies are recorded in:

```text
requirements-intel.txt
```

NVIDIA/CUDA dependencies are recorded in:

```text
requirements-nvidia.txt
```

## Training Data

The current dataset comes from the January 2019 Lichess rated-game database.

Filtering criteria:

- Rated Rapid games
- White Elo between 800 and 900
- Black Elo between 800 and 900
- BOT accounts excluded

The original compressed Lichess file is expected at:

```text
data/raw/lichess/lichess_db_standard_rated_2019-01.pgn.zst
```

The `data/` directory is intentionally excluded from Git.

## Generate the 5,000-Game Dataset

From the project root:

```bash
./scripts/data/filter_5000_games.sh
```

This produces:

```text
data/filtered/rapid_800_900_5000.pgn
```

Expected result:

```text
5,000 games
approximately 8 MB PGN
```

## Extract Training Positions

From the project root:

```bash
python3 scripts/data/extract_positions.py   data/filtered/rapid_800_900_5000.pgn   data/processed/rapid_800_900_positions_5000.csv
```

Current result:

```text
Processed 5,000 games
Extracted 235,358 positions
```

## Train the Model

Activate the environment:

```bash
source .venv/bin/activate
```

Start training:

```bash
python3 -m src.train
```

Current configuration:

```text
Batch size:     64
Learning rate:  0.001
Optimizer:      Adam
Loss:           CrossEntropyLoss
Maximum epochs: 15
```

Current 5,000-game run:

| Epoch | Test accuracy |
|------:|--------------:|
| 1 | 21.20% |
| 2 | 24.47% |
| 3 | 25.39% |
| 4 | 25.68% |
| 5 | **25.80%** |
| 6 | 25.66% |
| 7 | 25.21% |

Training was stopped after the model began to overfit.

The best checkpoint was therefore epoch 5.

## Inspection Tools

Inspect training examples:

```bash
python3 -m scripts.inspect.inspect_training_examples
```

Inspect the dataset:

```bash
python3 -m scripts.inspect.inspect_dataset
```

Inspect extracted positions:

```bash
python3 -m scripts.inspect.inspect_positions
```

## Baseline Evaluation

Run:

```bash
python3 -m scripts.benchmarks.evaluate_baseline
```

Previous baseline results:

```text
Random legal move:       ~7.13%
Most frequent move:      ~13.46%
500-game neural model:   ~17.9%
5000-game neural model:  25.80%
```

## Lucas Chess Benchmarks

Tournament PGNs are stored in:

```text
benchmarks/
```

Current results:

| HumanChess | Opponent | Wins | Draws | Losses | Score |
|---|---|---:|---:|---:|---:|
| 500-game model | Irina Monkey | 5 | 15 | 0 | 62.5% |
| 5,000-game model | Irina Monkey | 6 | 14 | 0 | 65.0% |
| 5,000-game model | Irina Horse | 0 | 4 | 16 | 10.0% |
| 5,000-game model | Irina Elephant | 0 | 2 | 18 | 5.0% |

This currently places HumanChess roughly around the Monkey level while Horse and Elephant remain substantially stronger.

## Development Direction

The next question is whether HumanChess should improve mainly through:

1. more training games,
2. a better neural-network architecture,
3. improved move selection,
4. limited tactical awareness.

Increasing the dataset from 500 to 5,000 games improved test accuracy from approximately:

```text
17.9% -> 25.8%
```

so additional training data remains a promising direction.

Benchmark games also show tactical problems such as missed mate threats and hanging pieces.

The next development stage will analyse these failures before deciding whether to scale the training set or add tactical knowledge.

## Design Goal

HumanChess is not intended to become a maximum-strength chess engine.

The objective is:

> Build a chess opponent that chooses moves like a real human player at a selected Elo level.

A successful HumanChess model should make good moves, ordinary moves, occasional poor moves, and realistic tactical mistakes at approximately the frequency expected from players in its target rating range.
