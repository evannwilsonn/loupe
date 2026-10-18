# Data

`scripts/get-deps.sh` downloads the binary version of CIFAR-10 into `data/cifar-10-batches-bin/`. It isn't committed.

The dataset has 60,000 colour photos, each 32×32 pixels, in ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

| Files | Contents | Used as |
|---|---|---|
| `data_batch_1.bin` … `data_batch_5.bin` | 50,000 training photos | 45,000 for training. The other 5,000 (500 per class, seed 2026) are held out for choosing between methods |
| `test_batch.bin` | 10,000 test photos, 1,000 per class | Scored once, at the end |

Each record is 3,073 bytes: one label byte, then 1,024 red, 1,024 green and 1,024 blue values, row by row.

**Citation.** Alex Krizhevsky. 2009. *Learning Multiple Layers of Features from Tiny Images.* Technical report, University of Toronto. The dataset page is https://www.cs.toronto.edu/~kriz/cifar.html
