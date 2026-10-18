# Loupe

A small convolutional network that sorts 32×32 colour photos into the ten CIFAR-10 classes. It is written and trained in C++ with LibTorch (the PyTorch C++ API) on two CPU cores, and it reaches **90.8% on the 10,000 test photos**. It is deployed two ways:
- a small C++ HTTP server;
- a browser demo that runs the same network in plain JavaScript.

Demo: `docs/index.html` (served with GitHub Pages).

Project 9 of my AI project run. Like the others, it follows seven steps: problem, data, baseline, model, evaluation, improvement, deployment.

## 1. Problem

Given a 32×32 photo, say which of ten things it shows: airplane, automobile, bird, cat, deer, dog, frog, horse, ship or truck. The classes are balanced, so the metric is plain accuracy, reported with a 95% Wilson interval. The constraint is the hardware: no GPU, two CPU cores, so the network has to be small and the training budget under an hour.

## 2. Data

CIFAR-10 (Krizhevsky, 2009): 50,000 training photos and 10,000 test photos, 1,000 per class in the test set. `scripts/get-deps.sh` downloads the binary version; it isn't committed. See [`data/README.md`](data/README.md).

- **Held-out set:** 500 photos per class (5,000 in all) are taken out of the training batches with a fixed seed. Every choice below was made on these: k, the learning rate, augmentation, epochs and test-time flipping.
- **Training set:** the other 45,000 photos.
- **Test set:** the 10,000 test photos, scored once at the end.
- **Normalisation:** per-channel mean and standard deviation, computed on the 45,000 training photos.

## 3. Baselines

Held-out accuracy, with the test score for comparison:

| Method | Held-out | Test |
|---|---|---|
| Guess one class | 10.0% | 10.0% |
| k-nearest neighbours on pixels (k = 3, chosen from 1–20) | 35.0% | 34.9% |
| Softmax regression on pixels (one linear layer, 20 epochs) | 40.6% | 39.9% |

The softmax learning rate was chosen from 0.0003, 0.001, 0.003 and 0.01. The smallest won, which sits at the edge of the grid, so a still smaller rate might do slightly better. Pixels alone don't say much: the nearest photo by colour is rarely the same kind of thing.

## 4. Model

A VGG-style CNN ([`src/loupe.cpp`](src/loupe.cpp)) with **288,746 weights**:

- Three stages. Each has two 3×3 convolutions with BatchNorm and ReLU, then a 2×2 max-pool. The widths are 32, 64 and 128, so the image goes 32×32 → 16×16 → 8×8 → 4×4.
- Then a global average pool and one linear layer to the ten classes.

**Training:**
- SGD with Nesterov momentum 0.9, weight decay 5e-4 and batch size 128.
- The learning rate warms up linearly to 0.1 over two epochs, then follows a cosine decay to zero.
- One epoch takes about 70 s on two cores.

## 5. Evaluation

| Run | Epochs | Held-out | Its own training photos |
|---|---|---|---|
| CNN, photos as given | 15 | 87.1% | 99.9% |
| CNN, augmented | 15 | 88.6% | 91.5% |
| CNN, augmented, longer | 40 | 90.8% | 95.8% |
| … plus its mirror image at test time | 40 | **91.8%** | |

**Test set, scored once:** **90.8%** (95% interval 90.2–91.4%), or 89.9% without the mirror image.

Per class on the test set, from best to worst:
- automobile 95.2%
- ship 95.1%
- truck 94.5%
- frog 92.9%
- horse 92.5%
- airplane 92.0%
- deer 91.8%
- dog 87.8%
- bird 87.1%
- cat 79.3%

The most common mistakes are 99 cats read as dogs and 70 dogs read as cats. Next come birds read as deer (34) and airplanes read as ships (32), which share sky and water backgrounds. The full confusion matrix is in `results/final.json` and on the page.

## 6. Improvement

Each step was decided on the held-out set:

1. **Augmentation.** Without it the network memorised: 99.9% on its own training photos but 87.1% on held-out ones. With random 4-pixel shifts (zero padding) and random mirroring it can't memorise exact pixels. Training accuracy fell to 91.5% and held-out accuracy rose to 88.6%, with the same network and the same 15 epochs.
2. **Longer training.** Augmented, it was still improving at epoch 15, so the final run is 40 epochs (45 minutes): 90.8% held-out.
3. **Test-time flipping.** Averaging the scores of each photo and its mirror image gave 91.8% held-out against 90.8%, so the final model uses it. On the test set it is worth about one point (89.9% to 90.8%).

## 7. Deployment

**C++ server** ([`src/serve.cpp`](src/serve.cpp), using cpp-httplib and stb_image):

```sh
./build/serve --port 8080
curl --data-binary @photo.jpg localhost:8080/classify
# {"top": [{"label": "airplane", "p": 0.711}, {"label": "ship", "p": 0.281}, ...], "ms": 8.0}
```

It takes any PNG or JPEG, centre-crops it to a square and shrinks it to 32×32 by averaging blocks of pixels. `GET /` serves the demo page.

**Browser:**
- `experiment final` folds each BatchNorm into the convolution before it, so inference is just convolution + ReLU, pooling, averaging and a linear layer.
- It checks the folded C++ loops against LibTorch: the largest logit difference is 1e-5, and they pick the same class on 200 of 200 photos.
- The page stores the folded weights as float16 (about 560 KB) and runs them with [`web/infer.js`](web/infer.js).
- `scripts/check-infer.mjs` runs that JavaScript on the first 1,000 test photos. It picks the same class as LibTorch on **1,000 of 1,000** (largest score difference 0.017), at about 90 ms per photo in Node.

On the page, the contact sheet's circles and crosses are the C++ model's answers on 120 test photos. Clicking a frame, or loading your own photo, runs the network in the browser.

Your own photos are harder than the test set. CIFAR photos are centred and tightly framed, and the network only knows ten classes, so anything else still gets one of them.

## Run it

```sh
./scripts/get-deps.sh                    # LibTorch 2.12.1 (CPU), CIFAR-10, httplib.h, stb_image.h
cmake -S . -B build && cmake --build build -j
./build/tests                            # 14 checks
./build/experiment baselines             # about 1 minute
./build/experiment train plain 15 0      # about 18 minutes each on two cores
./build/experiment train augment 15 1
./build/experiment train long 40 1       # about 45 minutes
./build/experiment final long            # scores the test set, exports the model
python3 scripts/build-demo.py            # writes web/weights.f16 and docs/index.html
node scripts/check-infer.mjs             # JavaScript vs LibTorch
python3 scripts/build-demo.py            # again, so the page shows the check
```

## Layout

```
src/loupe.hpp, loupe.cpp   data loading, split, augmentation, the network, folding, metrics
src/experiment.cpp         baselines, training runs, final evaluation and export
src/serve.cpp              the HTTP server
tests/tests.cpp            unit tests
web/infer.js               the network in JavaScript
web/template.html          the demo page (design notes in web/DESIGN.md)
web/weights.f16            the trained weights, folded, float16
results/                   every number in this README and on the page
```
