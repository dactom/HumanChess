# HumanChess Benchmarks

Tournament games used to compare HumanChess versions against Lucas Chess beginner engines.

## Test conditions

- 20 games per tournament
- Time control: 5+2
- HumanChess target strength: approximately 800–900 Elo human behaviour

## Results

| HumanChess model | Opponent | Wins | Draws | Losses | Score |
|---|---|---:|---:|---:|---:|
| 500 games | Irina Monkey | 5 | 15 | 0 | 62.5% |
| 5000 games | Irina Monkey | 6 | 14 | 0 | 65.0% |
| 5000 games | Irina Horse | 0 | 4 | 16 | 10.0% |
| 5000 games | Irina Elephant | 0 | 2 | 18 | 5.0% |

## Files

- `500games_vs_monkey.pgn`
- `5000games_vs_monkey.pgn`
- `5000games_vs_horse.pgn`
- `5000games_vs_elephant.pgn`

These benchmark games are used to evaluate whether changes to
training data, neural-network architecture, temperature, or tactical
penalties improve HumanChess without making it unrealistically strong.