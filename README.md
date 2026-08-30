# HumanChess

HumanChess is a neural-network chess engine designed to imitate the move choices of real human players rather than play the strongest possible chess.

Current target strength: **800-900 Elo human behaviour**

The model is trained using real Lichess games from players in this rating range.

---

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

HumanChess should return a legal move, for example:

```text
bestmove e7e5
```

Exit with:

```text
quit
```

---

## Using HumanChess in Lucas Chess

HumanChess works as a UCI chess engine.

In Lucas Chess, add a new external engine and point it to:

```text
/home/user/HumanChess/humanchess
```

The `humanchess` launcher starts the Python UCI engine from the correct project environment.

HumanChess has successfully played complete games and tournaments in Lucas Chess.

---

## Current Models

| Training set | Games | Positions | Best test accuracy | Best epoch |
|---|---:|---:|---:|---:|
| Initial | 500 | ~22,000 | ~17.9% | 10 |
| Expanded | 5,000 | 235,358 | 25.80% | 5 |
| January 2019 full | 19,159 | 908,404 | **30.55%** | 4 |

The current best model is:

```text
checkpoints/human_chess_policy_jan2019_all.pt
```

The previous 5,000-game model is:

```text
checkpoints/human_chess_policy_5000games.pt
```

The original 500-game model is:

```text
checkpoints/human_chess_policy_500games.pt
```

### January 2019 full-dataset split

```text
Total positions:    908,404
Training positions: 727,175
Test positions:     181,229
```

### January 2019 training result

| Epoch | Test accuracy |
|---:|---:|
| 1 | 26.88% |
| 2 | 29.61% |
| 3 | 30.20% |
| 4 | **30.55%** |
| 5 | 30.23% |
| 6 | 29.96% |

Training was stopped during epoch 7 because test accuracy had begun to fall while training loss continued decreasing.

The best checkpoint is therefore the epoch 4 model.

---

## Current Move Selection

The neural network produces scores for legal moves.

HumanChess then applies small tactical penalties and temperature-controlled sampling.

Current settings in:

```text
src/uci_engine.py
```

are:

```python
QUEEN_BLUNDER_PENALTY = 4.0
ROOK_BLUNDER_PENALTY = 1.5
TEMPERATURE = 0.5
```

These are deliberately soft penalties.

HumanChess is supposed to make mistakes because real 800-900 Elo players make mistakes.

The tactical layer is intended to reduce extreme mistakes without turning HumanChess into a conventional strong chess engine.

The basic move-selection pipeline is:

```text
800-900 Elo training games
        |
        v
neural-network move scores
        |
        v
legal-move filtering
        |
        v
soft tactical penalties
        |
        v
temperature-controlled sampling
        |
        v
HumanChess move
```

---

## Project Structure

```text
HumanChess/
├── src/
│   ├── baselines/
│   │   ├── __init__.py
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
│   │   ├── filter_january_2019_all.sh
│   │   └── filter_lichess.py
│   ├── inspect/
│   │   ├── inspect_dataset.py
│   │   ├── inspect_positions.py
│   │   └── inspect_training_examples.py
│   └── benchmarks/
│       └── evaluate_baseline.py
│
├── benchmarks/
│   ├── 500games_vs_monkey.pgn
│   ├── 5000games_vs_monkey.pgn
│   ├── 5000games_vs_horse.pgn
│   ├── 5000games_vs_elephant.pgn
│   ├── Jan2019_vs_monkey.pgn
│   └── README.md
│
├── data/
│   ├── raw/
│   ├── filtered/
│   └── processed/
│
├── checkpoints/
├── humanchess
├── requirements.txt
├── requirements-intel.txt
├── requirements-nvidia.txt
├── .gitignore
└── README.md
```

The following directories are excluded from Git because they may contain large generated files:

```text
/data/
/checkpoints/
/models/
/logs/
```

---

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

Intel Arc / XPU dependencies are recorded in:

```text
requirements-intel.txt
```

NVIDIA / CUDA dependencies are recorded in:

```text
requirements-nvidia.txt
```

---

## Training Data

The current dataset comes from the January 2019 Lichess standard rated-game database.

Filtering criteria:

- Rated Rapid games
- White Elo between 800 and 900
- Black Elo between 800 and 900
- BOT accounts excluded

The original compressed database file is expected at:

```text
data/raw/lichess/lichess_db_standard_rated_2019-01.pgn.zst
```

The `data/` directory is intentionally excluded from Git.

---

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

---

## Generate the Full January 2019 Dataset

From the project root:

```bash
./scripts/data/filter_january_2019_all.sh
```

This processes the entire January 2019 Lichess database without a 5,000-game limit.

Current result:

```text
19,159 matching games
```

The filtered PGN is:

```text
data/filtered/rapid_800_900_2019-01_all.pgn
```

---

## Extract Training Positions

### 5,000-game dataset

```bash
python3 scripts/data/extract_positions.py   data/filtered/rapid_800_900_5000.pgn   data/processed/rapid_800_900_positions_5000.csv
```

Current result:

```text
Processed 5,000 games
Extracted 235,358 positions
```

### Full January 2019 dataset

```bash
python3 scripts/data/extract_positions.py   data/filtered/rapid_800_900_2019-01_all.pgn   data/processed/rapid_800_900_positions_2019-01_all.csv
```

Current result:

```text
Processed 19,159 games
Extracted 908,404 positions
```

---

## Train the Model

Activate the environment:

```bash
source .venv/bin/activate
```

Start training:

```bash
python3 -m src.train
```

Current training configuration:

```text
Batch size:     64
Learning rate:  0.001
Optimizer:      Adam
Loss:           CrossEntropyLoss
Maximum epochs: 15
```

The dataset and checkpoint paths are configured near the top of:

```text
src/train.py
```

For the January 2019 full dataset, use:

```python
DATASET_PATH = (
    "data/processed/rapid_800_900_positions_2019-01_all.csv"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "human_chess_policy_jan2019_all.pt"
)
```

---

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

---

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
5,000-game neural model: 25.80%
January-all model:       30.55%
```

---

## Lucas Chess Benchmarks

Tournament PGNs are stored in:

```text
benchmarks/
```

Current results:

| HumanChess model | Opponent | Wins | Draws | Losses | Score |
|---|---|---:|---:|---:|---:|
| 500 games | Irina Monkey | 5 | 15 | 0 | 62.5% |
| 5,000 games | Irina Monkey | 6 | 14 | 0 | 65.0% |
| January-all | Irina Monkey | 5 | 15 | 0 | 62.5% |
| 5,000 games | Irina Horse | 0 | 4 | 16 | 10.0% |
| 5,000 games | Irina Elephant | 0 | 2 | 18 | 5.0% |

The January-all Monkey tournament showed that the much higher move-prediction accuracy did not yet produce a clear improvement in the 20-game Monkey result.

This is an important result because it suggests that prediction accuracy and practical playing strength are not the same thing.

---

## Benchmark Interpretation

The current strength picture is approximately:

```text
Irina Monkey
    |
    |  HumanChess is competitive here
    |
HumanChess
    |
    |  Horse is clearly stronger
    |
Irina Horse
    |
    |  Elephant is stronger again
    |
Irina Elephant
```

The current HumanChess model can compete with Monkey but still performs poorly against Horse and Elephant.

---

## Current Development Direction

Increasing the amount of training data produced a large improvement in move-prediction accuracy:

```text
500 games        -> ~17.9%
5,000 games      -> 25.80%
19,159 games     -> 30.55%
```

However, the January-all model did not clearly improve the 20-game Irina Monkey tournament result compared with the 5,000-game model.

This suggests that prediction accuracy is still improving with more training data, but actual playing strength may now be limited by other factors such as:

1. tactical blindness,
2. move-selection behaviour,
3. neural-network architecture,
4. limited ability to understand multi-move consequences.

The next stage should analyse benchmark games before deciding whether to:

- add more training data,
- modify the neural-network architecture,
- tune temperature and tactical penalties,
- or add limited tactical awareness.

---

## Design Goal

HumanChess is not intended to become a maximum-strength chess engine.

The objective is:

> Build a chess opponent that chooses moves like a real human player at a selected Elo level.

A successful HumanChess model should make good moves, ordinary moves, occasional poor moves, and realistic tactical mistakes at approximately the frequency expected from players in its target rating range.

The aim is not to eliminate mistakes.

The aim is to make the mistakes look human.
