# Table 4.5 (revised) — Per-LC indicators, burned pixels only

## (a) Test set, burned pixels only

Both analytical targets (target) and model predictions (pred) are shown.

|   LC code | LC name             |   n_burned |   sev_target |   per_target |   rec_target |   sev_pred |   per_pred |   rec_pred |   burned_prob |
|----------:|:--------------------|-----------:|-------------:|-------------:|-------------:|-----------:|-----------:|-----------:|--------------:|
|        10 | Forest              |       5561 |        0.592 |        0.594 |        0.252 |      0.597 |      0.602 |      0.260 |         0.688 |
|        11 | Orchards            |        554 |        0.210 |        0.301 |        0.152 |      0.204 |      0.305 |      0.157 |         0.444 |
|        20 | Shrubland           |       9652 |        0.635 |        0.627 |        0.500 |      0.644 |      0.630 |      0.508 |         0.635 |
|        30 | Grassland           |      18325 |        0.662 |        0.588 |        0.606 |      0.670 |      0.594 |      0.615 |         0.683 |
|        40 | Cropland            |       7308 |        0.345 |        0.570 |        0.472 |      0.352 |      0.577 |      0.485 |         0.513 |
|        41 | Covered Agriculture |         45 |        0.487 |        0.494 |        0.207 |      0.431 |      0.515 |      0.214 |         0.393 |
|        50 | Built Up            |        114 |        0.456 |        0.529 |        0.374 |      0.501 |      0.549 |      0.366 |         0.540 |
|        60 | Bare                |        895 |        0.480 |        0.664 |        0.415 |      0.511 |      0.671 |      0.429 |         0.580 |
|        61 | Fallow              |       1608 |        0.612 |        0.550 |        0.567 |      0.617 |      0.560 |      0.577 |         0.660 |
|        80 | Water               |        241 |        0.529 |        0.324 |        0.034 |      0.568 |      0.333 |      0.032 |         0.780 |

## (b) Full labeled sample, burned pixels only

Only analytical targets are shown; model predictions on training / validation pixels would be biased.

|   LC code | LC name             |   n_burned |   sev_target |   per_target |   rec_target |
|----------:|:--------------------|-----------:|-------------:|-------------:|-------------:|
|        10 | Forest              |      37914 |        0.593 |        0.598 |        0.255 |
|        11 | Orchards            |       3872 |        0.207 |        0.312 |        0.156 |
|        20 | Shrubland           |      64429 |        0.636 |        0.628 |        0.505 |
|        30 | Grassland           |     121880 |        0.662 |        0.586 |        0.606 |
|        40 | Cropland            |      48533 |        0.343 |        0.570 |        0.476 |
|        41 | Covered Agriculture |        284 |        0.389 |        0.421 |        0.185 |
|        50 | Built Up            |        821 |        0.453 |        0.539 |        0.325 |
|        60 | Bare                |       5783 |        0.481 |        0.654 |        0.409 |
|        61 | Fallow              |      10010 |        0.607 |        0.547 |        0.569 |
|        80 | Water               |       1480 |        0.558 |        0.343 |        0.035 |

## (c) Sample size comparison

|   LC code | LC name             |   n_burned_test |   n_burned_full |   ratio_test_full |
|----------:|:--------------------|----------------:|----------------:|------------------:|
|        10 | Forest              |            5561 |           37914 |             0.147 |
|        11 | Orchards            |             554 |            3872 |             0.143 |
|        20 | Shrubland           |            9652 |           64429 |             0.15  |
|        30 | Grassland           |           18325 |          121880 |             0.15  |
|        40 | Cropland            |            7308 |           48533 |             0.151 |
|        41 | Covered Agriculture |              45 |             284 |             0.158 |
|        50 | Built Up            |             114 |             821 |             0.139 |
|        60 | Bare                |             895 |            5783 |             0.155 |
|        61 | Fallow              |            1608 |           10010 |             0.161 |
|        80 | Water               |             241 |            1480 |             0.163 |

_Rounded to 3 decimals. Classes with n < 30 in the test set should be interpreted with caution or reported from the full sample._
