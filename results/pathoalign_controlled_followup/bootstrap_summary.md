# Bootstrap uncertainty summary

Twenty thousand seed-cluster bootstrap resamples were analyzed at threshold 0.50.

| N_u | Method | Median boundary | Lower 2.5% | Upper 97.5% | Above tested maximum |
|---:|---|---:|---:|---|---:|
| 750 | hybrid curriculum | 175 | 100 | above 225 | 0.10430 |
| 1500 | hybrid curriculum | 150 | 100 | above 225 | 0.18055 |
| 750 | pair consistency | 200 | 75 | above 225 | 0.23505 |
| 1500 | pair consistency | 125 | 100 | above 225 | 0.26635 |

The estimated observational-data effect was a median boundary reduction of 25 anchors for hybrid curriculum, with a 95% interval from -50 to 125 and probability 0.564378 of a positive reduction. Pair consistency had a median reduction of 50 anchors, an interval from -75 to 125, and probability 0.670121 of a positive reduction.
