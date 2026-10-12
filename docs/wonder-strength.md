# Wonder side strength

How strong is each wonder board side, and does that depend on the expansions and the number of
players? This report estimates every side's value in **victory points of final score margin** (own
score minus the opponents' mean) compared with an average side of the same table.

**Data:** 120,000 games (October 2026 rules): every combination of base, base + Leaders, base +
Cities, base + Leaders + Cities, base + Wonder Pack and all four together with 3 to 7 players, 4,000
games per table. Every seat is played by `board` (Hephaestus, `BoardAwareAgent`), the scripted agent
that values board powers explicitly. Raw fit: [`wonder-strength.json`](wonder-strength.json).

Regenerate with

```
python -m wonders.agents.wonder_values --all -g 4000 -p 4 -o docs/wonder-strength.json
```

## Method

1. **Random deals.** Boards and sides are drawn at random for every deal, so a side's value is
   identified by design (no selection by the players).
2. **Common random numbers.** Each deal is played twice with the same decks and the board assignment
   rotated around the table, so the luck of a deal is shared by the sides dealt in it.
3. **Regression.** A seat's margin is modelled as its own side's value minus the mean value of the
   opponents' sides. Each side has a main effect plus a deviation per table (expansion set × player
   count); the deviations are shrunk towards zero with a ridge penalty σ²/τ², where the spread τ of
   real table effects is estimated from the data (method of moments, one empirical Bayes step;
   here τ = 1.12 VP against a residual SD of σ = 9.8 VP per seat).
4. **Uncertainty.** Standard errors are cluster-robust by deal (the seats of a game and the two
   rotations share their noise). A per-table value has a standard error of 0.17 to 0.58 VP.

Values are centred on each table's sides: only differences between sides are identified, because
margins sum to zero. Averages over several tables therefore compare a side with different reference
sets (the Wonder Pack tables contain Manneken Pis B, which lowers everyone else a little).

## Overall

Each side's mean value over the tables it was dealt at (± one standard error).

| side | value (VP) | games |
|---|---:|---:|
| Manneken Pis B | +5.37 ± 0.11 | 8,352 |
| Great Wall A | +3.72 ± 0.12 | 8,442 |
| Rhodes B | +3.25 ± 0.06 | 32,478 |
| Ephesus B | +3.01 ± 0.06 | 32,394 |
| Olympia A | +1.67 ± 0.06 | 32,528 |
| Alexandria B | +1.63 ± 0.06 | 32,410 |
| Babylon B | +1.37 ± 0.06 | 32,538 |
| Byzantium A | +1.03 ± 0.08 | 14,144 |
| Giza B | +0.98 ± 0.06 | 32,400 |
| Olympia B | +0.91 ± 0.06 | 32,600 |
| Alexandria A | +0.89 ± 0.06 | 32,678 |
| Abu Simbel A | +0.59 ± 0.17 | 3,590 |
| Abu Simbel B | +0.06 ± 0.17 | 3,574 |
| Rhodes A | -0.01 ± 0.06 | 32,646 |
| Giza A | -0.08 ± 0.06 | 32,546 |
| Petra A | -0.38 ± 0.09 | 14,054 |
| Ephesus A | -0.50 ± 0.06 | 32,788 |
| Rome B | -0.57 ± 0.10 | 14,912 |
| Halicarnassus B | -0.76 ± 0.06 | 32,672 |
| Rome A | -1.08 ± 0.09 | 14,820 |
| Babylon A | -1.09 ± 0.06 | 32,592 |
| Manneken Pis A | -1.48 ± 0.12 | 8,530 |
| Stonehenge A | -1.81 ± 0.11 | 8,618 |
| Halicarnassus A | -2.00 ± 0.06 | 32,476 |
| Great Wall B | -2.61 ± 0.11 | 8,522 |
| Stonehenge B | -3.31 ± 0.11 | 8,596 |
| Byzantium B | -3.75 ± 0.09 | 13,950 |
| Petra B | -5.04 ± 0.09 | 14,150 |

## By expansions

Mean over 3 to 7 players. L = Leaders, C = Cities, W = Wonder Pack. Standard errors are about
0.10 to 0.18 VP.

| side | base | +L | +C | +L+C | +W | +L+C+W |
|---|---:|---:|---:|---:|---:|---:|
| Manneken Pis B |  |  |  |  | +5.2 | +5.1 |
| Great Wall A |  |  |  |  | +1.3 | +5.7 |
| Rhodes B | +3.6 | +2.6 | +3.0 | +2.2 | +4.1 | +2.3 |
| Ephesus B | +1.8 | +2.3 | +3.0 | +3.9 | +1.9 | +3.4 |
| Olympia A | +2.2 | +1.1 | +1.5 | +0.6 | +2.3 | +0.6 |
| Alexandria B | +0.1 | +1.3 | +1.1 | +3.1 | -0.2 | +2.7 |
| Babylon B | +0.0 | +0.3 | +2.1 | +2.0 | +0.5 | +1.8 |
| Byzantium A |  |  | +1.9 | +0.9 |  | +0.3 |
| Giza B | -0.3 | +0.7 | +0.7 | +1.4 | +0.2 | +1.6 |
| Olympia B | -0.3 | +0.4 | +0.8 | +1.8 | -0.3 | +1.4 |
| Alexandria A | -0.3 | +0.5 | +0.4 | +1.4 | -0.1 | +1.8 |
| Abu Simbel A |  |  |  |  |  | +0.6 |
| Abu Simbel B |  |  |  |  |  | +0.1 |
| Rhodes A | -0.1 | -0.5 | -0.2 | -0.8 | +0.6 | -0.7 |
| Giza A | +0.2 | -0.8 | -0.2 | -1.0 | +0.4 | -0.7 |
| Petra A |  |  | -0.3 | -0.6 |  | -0.3 |
| Ephesus A | -1.3 | -0.8 | -1.0 | -0.2 | -1.1 | -0.3 |
| Rome B |  | -1.1 |  | -0.3 |  | -0.8 |
| Halicarnassus B | -1.3 | -1.0 | -1.3 | -0.6 | -1.1 | -0.9 |
| Rome A |  | -1.2 |  | -1.2 |  | -1.2 |
| Babylon A | -1.5 | -1.7 | -1.3 | -1.3 | -1.1 | -1.2 |
| Manneken Pis A |  |  |  |  | -1.8 | -1.6 |
| Stonehenge A |  |  |  |  | -1.8 | -2.3 |
| Halicarnassus A | -2.8 | -2.2 | -2.5 | -1.9 | -2.3 | -1.9 |
| Great Wall B |  |  |  |  | -3.7 | -2.0 |
| Stonehenge B |  |  |  |  | -3.0 | -4.1 |
| Byzantium B |  |  | -3.1 | -3.8 |  | -4.4 |
| Petra B |  |  | -4.6 | -5.4 |  | -5.2 |

## By player count

Mean over the expansion combinations. Standard errors are about 0.1 to 0.2 VP (0.2 to 0.5 for the
Wonder Pack boards, which appear in fewer tables).

| side | 3p | 4p | 5p | 6p | 7p |
|---|---:|---:|---:|---:|---:|
| Manneken Pis B | +3.5 | +6.3 | +5.5 | +5.4 | +4.9 |
| Great Wall A | +3.9 | +4.1 | +3.6 | +2.5 | +3.4 |
| Rhodes B | +4.7 | +2.6 | +2.7 | +2.6 | +2.3 |
| Ephesus B | +0.4 | +3.3 | +3.3 | +3.2 | +3.5 |
| Olympia A | +2.2 | +1.1 | +1.2 | +1.2 | +1.4 |
| Alexandria B | +3.4 | +1.4 | +0.8 | +0.4 | +0.8 |
| Babylon B | +0.2 | +2.5 | +1.2 | +0.9 | +0.7 |
| Byzantium A | -0.3 | +1.6 | +1.1 | +1.3 | +1.3 |
| Giza B | +0.8 | +0.4 | +0.6 | +0.9 | +0.9 |
| Olympia B | +2.4 | +0.1 | +0.5 | -0.0 | +0.2 |
| Alexandria A | +1.2 | +0.7 | +1.0 | +0.1 | +0.2 |
| Abu Simbel A | -0.1 | +2.3 | +0.7 | +0.0 | +0.0 |
| Abu Simbel B | +0.9 | +0.7 | +0.2 | -0.6 | -0.9 |
| Rhodes A | +1.1 | -0.7 | -0.6 | -0.4 | -0.9 |
| Giza A | +0.2 | -1.8 | -0.3 | +0.1 | +0.2 |
| Petra A | -0.7 | -0.5 | -0.4 | -0.2 | -0.2 |
| Ephesus A | -1.8 | -1.0 | -0.4 | -0.3 | -0.4 |
| Rome B | -3.4 | -0.1 | -0.0 | +0.1 | -0.2 |
| Halicarnassus B | -2.7 | -0.8 | -0.8 | -0.8 | +0.0 |
| Rome A | -2.3 | -0.9 | -1.3 | -0.6 | -1.1 |
| Babylon A | +0.6 | -1.9 | -1.6 | -1.6 | -2.2 |
| Manneken Pis A | -1.4 | -2.1 | -1.7 | -1.6 | -1.7 |
| Stonehenge A | -2.2 | -2.1 | -1.7 | -2.0 | -2.3 |
| Halicarnassus A | -3.1 | -2.3 | -2.0 | -1.9 | -2.1 |
| Great Wall B | -3.5 | -2.9 | -2.6 | -3.0 | -2.2 |
| Stonehenge B | -2.4 | -3.0 | -4.8 | -3.8 | -3.6 |
| Byzantium B | -3.9 | -3.3 | -4.3 | -3.9 | -3.4 |
| Petra B | -7.1 | -5.1 | -5.1 | -3.9 | -4.1 |

## Findings

* **Sides differ by up to 10 VP.** The spread between the strongest and the weakest side (Manneken
  Pis B +5.4, Petra B −5.0) is about one residual standard deviation of a seat's margin, so the
  deal matters a lot for a single game and a great deal for ratings over few games.
* **Side B is not always the strong side.** B beats A for Rhodes, Ephesus, Alexandria, Babylon,
  Giza, Halicarnassus, Rome and the Manneken Pis; A is stronger for Olympia, Byzantium, Petra, the
  Great Wall and Stonehenge (and slightly for Abu Simbel, +0.5 ± 0.24).
* **Expansions shift values by several VP.** Alexandria B goes from +0.1 (base) to +3.1 (Leaders +
  Cities), Babylon B gains about 2 VP with Cities, Olympia A loses about 1.6 VP with Leaders + Cities,
  and the Great Wall A rises from +1.3 (with the base game only) to +5.7 (with Leaders and Cities).
* **Player count matters most at three players.** At 3p Rhodes B (+4.7), Alexandria B (+3.4) and
  Olympia B (+2.4) peak, while Ephesus B (+0.4), Rome B (−3.4) and Petra B (−7.1) are at their worst.
  From 4 players on most values change by less than 1 VP (Babylon B and Stonehenge B are
  exceptions).

## Caveats

* **The numbers describe the agent as much as the boards.** Petra B's second stage costs 14 coins;
  in a 60-game check with `board` in every seat it was built in only 18 games, so the side's −5 VP
  is to a large part an agent that does not save up for it. Other sides with unusual powers
  (Stonehenge B, Byzantium B, the Great Wall's stage order) may be undervalued for the same reason.
  The bundled network refuses the Wonder Pack boards, so a check with the network covers the other
  boards only.
* **Manneken Pis B** (one stage for one of each of the seven resources: 7 coins, 7 VP and military,
  plus 4 starting coins) comes out as the strongest side by a wide margin; worth a rules check and a
  second agent before drawing conclusions.
* Values are relative to the sides at the same table and to `board`'s play against copies of itself;
  stronger or weaker opponents can change them.

## Uses

* Board-adjusted ratings: the GUI's Elo could credit the expected margin of the boards dealt.
* Balanced deals or board-adjusted margins for benchmarks, which reduces seed variance.
* The expansion builder could report a new board's value next to the official ones.
* Agent fixes where a side looks weak because the agent misplays it.
