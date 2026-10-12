# Card and board strength

How many victory points is a card or a wonder board worth? The expansion builder answers in three
steps, from instant to slow, all in the same unit: **victory points of final score margin** (own score
minus the opponents' mean).

* A **card** is valued relative to discarding it for coins: a card worth building has a positive
  value, and −1 means "no better than the 3 coins of a discard".
* A **board side** is valued relative to an average official side, the scale of
  [wonder-strength.md](wonder-strength.md).

The code is `wonders/agents/strength.py` (engine side) and `wonders/play/strength.py` (builder side:
stored results, background jobs, the overview).

## 1. The rate card (while editing)

The rate card values an item from its effects, cost and Age in a few milliseconds. Its features are
measured through the uniform effect hooks in **reference positions**: the tables of a few simulated
games (four per player count, `board` agents) after each Age and at the end.

| feature | how it is measured | prior weight |
|---|---|---|
| points | VP the effects score in the final positions (counting cards and guilds included) | 1 |
| coins | coins when built (in the position after the item's Age) / 3 | 1 |
| on-build effects | change of the score margin when plunder, debt, tokens or grants are applied to copies of those positions | 1 |
| military | change of the margin from the remaining conflicts when the shields are added | 1 |
| science | change of the final science score | 1 |
| production | resource slots × Ages they are used in (a choice counts a little more) | 0.6 |
| trade | resources (per side) traded cheaper × Ages | 0.25 |
| chains | later cards it makes free; whether it is free through a chain | 1, 0.5 |
| cost | resource units; coins / 3 | −0.4, −1 |
| powers | board and leader powers by hook (Babylon's last card, Olympia's free build, discard builds, discounts, …), priors per use or per Age | 1 |
| turn | the play itself: the coins a discard would have given / 3 | −1 |

The priors come from the rules' exchange rates: 3 coins are 1 VP, a discard gives 3 coins, a
resource bought from a neighbour costs 2 coins. The bundled calibration (`wonders/agents/data/strength.json`,
step 4) replaces them with weights fitted to simulations of the official cards.

A board's raw rate is the sum of its stages' rates (each weighted by how often stages at that position
are built), its starting resource, extra coins and board powers. A scale fitted to the measured
strengths of the official sides turns it into VP relative to an average side. A stage that copies a
neighbour's stage counts as the official average of that position.

Meta rules of the expansion (coins per point, discard coins, military tokens) feed into the features.

## 2. One item, simulated (after saving)

After a save, each card and board whose definition changed is simulated in the background
(`simulate_card`, `simulate_wonder`):

* **Card:** a game of the base game is played with `board` agents. At a random turn of the card's Age,
  the card replaces a card in a random seat's hand, and the game is played on **twice from that
  position**: once building the card, once discarding it. Both runs share the deal and every agent's
  random seed up to that point, so the luck of the deal cancels (common random numbers). The value is
  the mean margin difference over 40 such pairs; positions where the card cannot be paid are skipped.
* **Board:** the same deal is played twice with seat 0 on the new side and on a random official side;
  the margin difference plus that official side's measured strength is one sample (30 per side).

The standard error is about ±0.5 VP for Age III cards and ±1.5 VP for Age I cards, whose effect has a
whole game to diverge in.

## 3. The whole expansion (on request)

"Simulate the whole expansion" (`simulate_expansion`) plays games with the expansion's cards in the
decks. In each watched game, every new card that some seat could build gets one paired sample at a
moment where it naturally came up (build vs discard), so cards are measured among each other, with
their combinations. Boards are paired against official sides with the expansion in play.

The results then **recalibrate the rate card for this expansion** (`recalibrate`): a straight line
`value ≈ offset + slope × rate` fitted to the expansion's simulated cards, each weighted by its
precision, and shrunk towards `(0, 1)` so that a few cards move it a little and many tight estimates
move it a lot. While editing afterwards, the rate card shows the corrected value.

The overview's "estimate" column takes the best available value: a fresh expansion simulation, a
fresh item simulation, or the (recalibrated) rate card. An item edited after its simulation is shown
as changed until it is simulated again (results are keyed by a hash of the item's definition, its
colours and the meta rules).

## 4. Calibration on the official cards

`python -m wonders.agents.strength -g 600 -p 4` (`--refit` reuses the stored simulations) simulates the official Age cards (base game and
Cities decks) with natural pairs, then fits the rate card's weights by weighted ridge regression
shrunk towards the priors (`fit_weights`), and the board scale to the measured side strengths. It
writes `wonders/agents/data/strength.json` with the weights, every official card's simulated value
(the comparison ranges the builder shows) and the leave-one-out error of the rate card.

**Current calibration** (October 2026, 600 watched games each with the base game and with Cities, 117
cards, median 955 paired samples and ±0.25 VP per card):

| feature | prior | fitted |
|---|---:|---:|
| points | +1.00 | +0.74 |
| coins | +1.00 | +1.07 |
| on_build | +1.00 | +1.01 |
| military | +1.00 | +0.72 |
| science | +1.00 | +0.64 |
| production | +0.60 | +1.11 |
| trade | +0.25 | +0.39 |
| chain_out | +1.00 | +0.40 |
| chain_in | +0.50 | +0.21 |
| resource_cost | -0.40 | -0.05 |
| coin_cost | -1.00 | -0.81 |
| powers | +1.00 | +0.64 |
| turn | -1.00 | -1.00 |
| intercept | 0 | +0.28 |

* **Cards:** the simulated values spread with an SD of 1.42 VP; the rate card predicts each
  official card, fitted without it, to 0.94 VP (leave-one-out RMSE), so it explains a
  little over half of the differences between cards (R² ≈ 0.56). Expect ±1 VP on a new card
  before simulation.
* **Boards:** the raw board rate correlates with the measured side strengths at r = 0.52
  (RMSE 1.9 VP after scaling): a rough first guess. Board powers and stage timing are
  where the simulation matters most.
* Strongest official cards relative to a discard: Timber Yard (+5.8), Builders Guild (+5.5), Mercenaries (+5.1), Palace (+4.8), Tree Farm (+4.7); weakest:
  Embassy (-2.9), Guardhouse (-1.2), Consulate (-0.5), Architect Firm (-0.2), Residence (-0.2).
* Notable fitted weights: resource costs barely matter once a card can be built (the simulation values
  a card *when it is affordable*), coin costs a little less than 3 coins per VP, and production almost
  twice the prior, as early resources save trading for the rest of the game. The turn weight stays at
  its prior: every card has the same turn feature, so the intercept takes up any difference.

## Caveats

* Every value describes the item **as the `board` agent plays it**. A power the agent does not plan
  for (saving coins for an expensive stage, timing a chain) is undervalued; see the Petra B example in
  [wonder-strength.md](wonder-strength.md).
* Paired values measure the item **when it can be built**: an expensive card that is rarely affordable
  has a fair value per build but is built less often. The rate card's cost terms and the "how often
  built" column of the balance check cover that side.
* The rate card is linear: strong interactions (a card that only shines next to another new card) show
  up only in the simulation of the whole expansion.

## References

* Rosen, S. (1974). Hedonic prices and implicit markets. *Journal of Political Economy*, 82(1), 34–55
  (valuing a good by its features).
* Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, ch. 4 (common random numbers
  and other variance reduction). Springer.
* Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression. *Technometrics*, 12(1), 55–67.
* Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*.
  Cambridge University Press (partial pooling, as in the recalibration).
