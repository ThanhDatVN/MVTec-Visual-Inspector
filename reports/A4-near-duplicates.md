# A4 — Near-duplicate captures across splits (rule L6)

All twelve VisA categories, official one-class split, validation carved from train (15%, seed 0).
`scripts/near_duplicate_audit.py`; generated table and per-image lists in
[studies/a4/](studies/a4/near_duplicates.md).

## Method

On a fixed capture rig every photograph of a part looks alike, so a fixed similarity threshold
either flags everything or nothing. Each criterion is therefore calibrated per category: for every
training image, the distance to its nearest *other* training image describes how close two
distinct captures normally are; a validation or test image closer to the training set than the
1st percentile of that distribution is a near-duplicate candidate. Under exchangeability about 1%
of images fall below that line by construction, so a leak shows up as an *excess* over ~1%.

## Results

| category | pHash nn median / p1 (train) | pHash exact matches val / test | thumbnail RMSE nn median / p1 (train) | val below p1 | test below p1 | pixel-identical val / test |
|---|---|---|---|---|---|---|
| `candle` | 6 / 2 | 2 / 6 | 9.0 / 6.1 | 2/135 | 3/200 | **1** / 0 |
| `capsules` | 12 / 8 | 0 / 0 | 28.3 / 26.3 | 0/81 | 0/160 | 0 / 0 |
| `cashew` | 2 / 0 | 14 / 36 | 10.9 / 8.9 | 0/68 | 1/150 | 0 / 0 |
| `chewinggum` | 2 / 0 | 17 / 34 | 11.6 / 6.5 | 0/68 | 0/150 | 0 / 0 |
| `fryum` | 4 / 0 | 0 / 1 | 15.0 / 9.4 | 1/68 | 1/150 | 0 / 0 |
| `macaroni1` | 4 / 1 | 2 / 4 | 11.8 / 8.5 | 1/135 | 1/200 | 0 / 0 |
| `macaroni2` | 10 / 4 | 1 / 0 | 33.2 / 23.7 | 2/135 | 0/200 | 0 / 0 |
| `pcb1` | 2 / 0 | 51 / 44 | 11.6 / 9.1 | 2/136 | 3/200 | 0 / 0 |
| `pcb2` | 2 / 0 | 14 / 13 | 10.5 / 7.8 | 3/135 | 1/200 | 0 / 0 |
| `pcb3` | 0 / 0 | 58 / 69 | 10.1 / 7.7 | 2/136 | 0/201 | 0 / 0 |
| `pcb4` | 0 / 0 | 105 / 103 | 10.1 / 7.1 | 3/136 | 1/201 | 0 / 0 |
| `pipe_fryum` | 3 / 0 | 11 / 24 | 7.3 / 3.5 | 0/68 | 0/150 | 0 / 0 |

Thumbnails are 64×64 grey; RMSE is in grey levels; RMSE < 1 counts as pixel-identical.

## Reading

- **No leak between training and test.** On every category the share of test images closer to
  the training set than the training set's own 1st percentile is 0–1.5%, the rate expected by
  construction; no test image is pixel-identical to a training image. Rule L6 passes.
- **One duplicate capture inside the original training pool:** a `candle` validation image is
  pixel-identical to a training image. Validation is carved from train, so this is a repeated
  capture in VisA's training split, not a train/test leak; its effect on the threshold is at most
  one calibration score.
- **The 64-bit pHash of the original rule is unusable on these rigs.** Distinct captures collide
  at Hamming distance 0 (`pcb4`: median nearest-neighbour distance 0 inside train; hundreds of
  exact matches across splits, none pixel-identical). A pHash-based L6 check would either flag a
  third of every PCB split or, with a looser bar, nothing; the thumbnail criterion replaces it as
  the audit of record.
