
# Assignment 2 Report

This report compares the corrected models (AlexNet, VGG16, ResNet18) across all four
datasets (cells, chest, lesions, orgs). It reports the dataset profiles, the final chosen
configuration, the full metrics for every dataset/model pair.

---

# Task 1: Pipeline Reconstruction

## 1. Dataset Profiles

The four datasets differ in channels (grayscale vs RGB), number of classes, and size.
These differences are the main reason different models win on different datasets, so they
are listed first.

| Dataset | Channels | Classes | Train size | Test size | Character |
|---|---|---|---|---|---|
| cells | 3 (RGB) | 8 | 13,671 | 3,421 | Colour, many samples per class  |
| chest | 1 (grayscale) | 2 | 5,232 | 624 | Binary, smallest set |
| lesions | 3 (RGB) | 7 | 8,010 | 2,005 | Colour, imbalanced classes |
| orgs | 1 (grayscale) | 11 | 15,367 | 8,216 | Most classes, largest set  |
---

## 2. Final Benchmark Results
These are the results with this hyperparameters : Epochs : 20 , Learning Rate : 0.001 Droptout : 0.2 \
All values are on the test set. Best model per dataset in bold.

| Dataset | Model | Accuracy | Precision | Recall | Macro F1 |
|---|---|---|---|---|---|
| cells | AlexNet | 96.4% | 97.1% | 95.6% | 96.2% |
| cells | **VGG16** | **97.2%** | **96.9%** | **97.4%** | **97.1%** |
| cells | ResNet18 | 96.7% | 96.6% | 96.2% | 96.3% |
| chest | AlexNet | 85.9% | 90.0% | 81.5% | 83.5% |
| chest | **VGG16** | **88.6%** | **91.0%** | **85.3%** | **87.1%** |
| chest | ResNet18 | 82.4% | 88.3% | 76.7% | 78.6% |
| lesions | AlexNet | 75.3% | 49.3% | 44.1% | 45.5% |
| lesions | VGG16 | 74.2% | 56.6% | 38.8% | 40.0% |
| lesions | **ResNet18** | **76.0%** | **52.2%** | **47.5%** | **48.6%** |
| orgs | AlexNet | 90.5% | 89.9% | 89.3% | 89.5% |
| orgs | VGG16 | 91.6% | 91.1% | 90.4% | 90.5% |
| orgs | **ResNet18** | **92.2%** | **91.6%** | **90.9%** | **91.0%** |

### Accuracy targets

| Dataset | Required | Achieved (best model) | Status |
|---|---|---|---|
| cells | 90% | 97.2% (VGG16) | Passed |
| chest | 87% | 88.6% (VGG16) | Passed |
| lesions | 67% | 76.0% (ResNet18) | Passed  |
| orgs | 83% | 92.2% (ResNet18) | Passed |

---

## 3. Model Comparison per Dataset
Why the recommended model beats the other two.

### cells: VGG16

*Profile: RGB, **8 classes**, **13,671 train (2nd largest set)**.*

| Model | Accuracy | Macro F1 |
|---|---|---|
| **VGG16** | **97.2%** | **97.1%** |
| ResNet18 | 96.7% | 96.3% |
| AlexNet | 96.4% | 96.2% |

All three score within ~1%. With colour images and plenty of samples per class, cells is
easy and every model saturates near the top accuracy and F1 stay close, confirming the
classes are well balanced and separable. VGG16 edges ahead on macro F1 because its stack of
small 3×3 convolutions captures fine cell texture slightly better. Honest reading: the gap
is real but small; VGG16 is chosen because it leads on every metric.

### chest: VGG16

*Profile: grayscale, **2 classes (binary, fewest)**, **5,232 train (smallest set)**.*

| Model | Accuracy | Recall | Macro F1 |
|---|---|---|---|
| **VGG16** | **88.6%** | **85.3%** | **87.1%** |
| AlexNet | 85.9% | 81.5% | 83.5% |
| ResNet18 | 82.4% | 76.7% | 78.6% |

VGG16 is the only model above the 87% target and wins decisively on recall (85.3% vs
ResNet18's 76.7%) it misses far fewer positive cases, which matters most in medical
screening. Why does the deeper ResNet18 lose here? chest is the simplest task (binary)
with the *fewest* training images (5,232). ResNet18's depth is overkill for this, so the
extra capacity adds variance and overfits instead of helping its low recall shows it
missing positives. The simpler VGG16 matches the task and generalizes better.

### lesions: ResNet18

*Profile: RGB, **7 classes (most imbalanced)**, **8,010 train**.*

| Model | Accuracy | Macro F1 |
|---|---|---|
| **ResNet18** | **76.0%** | **48.6%** |
| AlexNet | 75.3% | 45.5% |
| VGG16 | 74.2% | 40.0% |

lesions is the key dataset of the report. For every model, accuracy (~76%) is far above
macro F1 (~49%). Across 7 classes, that large gap is the signature of class imbalance:
one class dominates, so a model scores high accuracy by leaning on the majority class while
doing poorly on the minority ones.

ResNet18 is recommended because it has the best macro F1 (48.6%) the most balanced
performance across the 7 classes, not just the best majority-class score. Its residual
connections help it learn the subtle minority-class boundaries that VGG16 (worst F1, 40.0%)
misses despite similar accuracy.

### orgs: ResNet18

*Profile: grayscale, **11 classes (most)**, **15,367 train (largest set)**.*

| Model | Accuracy | Macro F1 |
|---|---|---|
| **ResNet18** | **92.2%** | **91.0%** |
| VGG16 | 91.6% | 90.5% |
| AlexNet | 90.5% | 89.5% |

orgs has the most classes and the most data, and it is balanced (accuracy ≈ F1, unlike
lesions). ResNet18 wins on every metric. Why the deepest model wins here: with 11
classes the decision boundaries are harder, and with 15,367 training images there is enough
data to justify the extra capacity. The residual connections keep gradients flowing cleanly
through the deeper network, so it fits the multi-class structure better than VGG16 or
AlexNet without overfitting.

### Cross-dataset pattern

chest and orgs are both grayscale, yet ResNet18 loses one and wins the other  and the
dataset profiles explain exactly why:

| | chest | orgs |
|---|---|---|
| Classes | 2 (binary) | 11 |
| Train size | 5,232 (smallest) | 15,367 (largest) |
| ResNet18 result | Worst (82.4%) | Best (92.2%) |

On chest, the problem is too simple and the data too small, so ResNet18's depth overfits.
On orgs, the problem is hard enough and the data large enough that the same depth pays off.
Conclusion: match model capacity to both task complexity (number of classes) and dataset
size. Deeper is not always better (chest), and simpler is not always enough (orgs).

---

## 4. Architectural Recommendations

For each dataset we recommend the model with the best benchmark results, based on the
dataset profile (Section 1) and the observed metrics (Section 3).

| Dataset | Recommended Model | Best acc / F1 | Why this model |
|---|---|---|---|
| cells | VGG16 | 97.2% / 97.1% | Easy RGB set (~1,700 img/class); all models >96%, so decided by small margins. VGG16 leads on every metric, its stacked 3×3 convs capture the fine cell texture best. |
| chest | VGG16 | 88.6% / 87.1% | Only model above the 87% target, and best recall (85.3%) so it misses the fewest positives, key for medical screening. ResNet18 is worst (82.4%) because its depth overfits this small (5,232 img) binary task. |
| lesions | ResNet18 | 76.0% / 48.6% | Imbalanced 7-class set: high accuracy but low F1, so macro F1 is the fair metric. ResNet18 has the best macro F1 = most balanced across classes; its skip connections catch the minority classes VGG16 misses (F1 only 40.0%). |
| orgs | ResNet18 | 92.2% / 91.0% | Most complex (11 classes) and largest set (15,367 img), so the extra depth is justified and wins every metric. Residual connections let gradients flow cleanly, making the deep network easy to train without overfitting. |

Overall: VGG16 for the simpler datasets (cells, chest); ResNet18 for the complex ones
(lesions, orgs). Guiding principle: match model capacity to both task difficulty (number
of classes) and dataset size, depth pays off only when the problem is hard enough to need
it and large enough to train it (orgs), while a lighter model wins when the task is simple
and the data is limited (chest).

---

## 5. Experiments (Hyperparameter Tuning)

Each configuration change was tested one factor at a time so its effect could be
attributed cleanly. This section shows what changed and why the final value was kept.

### Shuffle before the split (bug fix)

The recovered `data.py` took the last 10% of samples as validation without shuffling.
Because these medical datasets are grouped by class, the validation set was not
representative and training could miss entire classes.

**Before (no shuffle) vs after (shuffle), lesions / ResNet18:**

| | Accuracy |
|---|---|
| Before (no shuffle) | 54.2% |
| After (shuffle) | 76.0% |

Before the fix, validation accuracy also swung wildly between epochs (e.g. 37% → 94%),
because the val set didn't match the training distribution. After shuffling, validation
tracked test performance smoothly. This is a correctness fix, not just tuning.

### Learning rate: 0.0003 vs 0.001 vs 0.002

| LR | Effect | Evidence (cells / VGG16 accuracy) |
|---|---|---|
| 0.0003 | Under-trains in the epoch budget | 79.3% (AlexNet, 10 epochs)  underfit |
| 0.001 | Converges cleanly | 97.2% |
| 0.002 | Overshoots / destabilizes | 93.9% (VGG16 drops from 97.2%) |

At 0.0003 the model needs many more epochs to converge (inefficient). At 0.002 the steps
are too large training becomes unstable and VGG16 (no residual connections to smooth
gradients) degrades noticeably. 0.001 is the balance point.

### Dropout: 0.5 vs 0.2

Comparison of the harder datasets when moving from dropout 0.5 to 0.2:

| Metric | Dropout 0.5 | Dropout 0.2 | Change |
|---|---|---|---|
| lesions best macro F1 | 42.3% | **48.6%** | +6.3 |
| orgs best accuracy | 90.9% | **92.2%** | +1.3 |

Dropout 0.5 was too aggressive for the harder datasets it removed too much capacity,
so the models couldn't fit the difficult boundaries. Lowering to 0.2 restored that capacity
and improved lesions and orgs, while the easy dataset (cells) was already saturated and
barely changed. Note dropout only affects AlexNet and VGG16 (their classifier heads);
ResNet18 has no dropout layer, so its gains here come mainly from the extra epochs.

### Epochs: 10 vs 15 vs 18 vs 20

10 epochs left the harder datasets under-trained (especially at low LR). Increasing to 20
let lesions and orgs converge, while validation curves flattened by ~epoch 18–20 (no
further gain and no overfitting), so 20 was kept rather than pushing higher.

### Summary of all runs

| Run | Shuffle | LR | Epochs | Dropout | Outcome |
|---|---|---|---|---|---|
| A | No | 0.001 | 10 | 0.5 | Non-representative val split; lesions/ResNet collapsed to 54%. |
| B | Yes | 0.0003 | 10 | 0.5 | Underfitting, LR too low for the epoch budget. |
| C | Yes | 0.0003 | 18 | 0.5 | Strong; passes all targets. Close second, but slower LR and weaker chest. |
| D | Yes | 0.001 | 15 | 0.5 | Good, but dropout 0.5 over-regularizes the harder datasets. |
| **E (final)** | **Yes** | **0.001** | **20** | **0.2** | **Best balance; meets every target; matches committed config.json.** |
| F | Yes | 0.002 | 20 | 0.2 | LR too high training destabilized (cells/VGG16 fell to 93.9%). |
---
# Task 2: Green Initiative

## What we were asked to do

The board's message was simple: the original models work, but they are heavy. VGG16 and
ResNet18 each carry about 11 million parameters, which is huge for diagnostic devices that
need to run fast and cheap. So design a lighter model that keeps roughly
the same accuracy as the originals while using far less compute, memory, and energy.

---

## 1. Architectural Downscaling

We started from ResNet18 because it was the most reliable model in Task 1, and its skip
connections make it easy to shrink without breaking training. We reduced it in two stages.

### Step 1: Cut the width in half

ResNet18 runs at 64 -> 128 -> 256 -> 512 channels across its four stages. We halved every one of
these to 32 -> 64 -> 128 -> 256. Because a convolution's size depends on
`input channels x output channels`, halving the width makes each layer about 4x smaller.
This took the model from ~11.2M parameters down to ~2.8M.

Result (accuracy stayed close to the full models, at ~4x smaller):

| Dataset | Half-width GreenResNet | Params |
|---|---|---|
| cells | 96.8% (F1 96.4) | 2.80M |
| chest | 82.4% (F1 78.7) | 2.79M |
| lesions | 75.3% (F1 49.1) | 2.80M |
| orgs | 92.8% (F1 92.1) | 2.80M |

The accuracy held up well - on cells, lesions and orgs it was right next to the full models even
at a quarter of the size. That is what convinced us there was still slack in the model and we
could shrink it further.

### Step 2: Add bottleneck blocks on top

On top of the narrower width, we rebuilt each block as a bottleneck. Instead of running the
expensive 3x3 convolution on the full number of channels, a bottleneck first uses a cheap 1x1
convolution to squeeze the channels down to a quarter, runs the 3x3 on that small squeezed
version, and then uses another 1x1 convolution to expand back to full size. The block still
takes in and puts out the same number of channels, but the costly part in the middle only ever
touches a quarter of them.

Added on top of the half width, it shrank the model again, down to about 224K parameters.

### The two changes stack up

Our final model uses both changes together.

| Model | Parameters | Size vs ResNet18 |
|---|---|---|
| ResNet18 (original) | 11,172,936 | 1x |
| Step 1: half width | 2,797,096 | ~4x smaller |
| Step 2: half width + bottleneck | 223,976 | ~50x smaller |

Halving the width gives ~4x, and the bottleneck on top gives roughly another ~14x on the
expensive deep blocks - about a 50x reduction overall, while keeping the same depth and the
residual connections that make the network train well.

---

## 2. Efficiency Verification Matrix

For every dataset-model pair, `benchmark_green.py` now records the three things the assignment
asks for:

1. **Total training runtime** - measured with a timer around the whole training loop.
2. **Inference latency per sample** - we run one warm-up batch (not timed), then time the full
   test set and divide by the number of images, giving milliseconds per image.
3. **Peak memory** - measured separately for training and for inference. Before the inference
   measurement we free the optimizer state and gradients and empty the cache, so the inference
   number reflects inference alone and isn't inflated by leftover training memory.

We also log the parameter count, accuracy, and macro F1 so accuracy and cost sit side by side.

The matrix below is the **final half-width bottleneck GreenResNet** measured against the three
original models, all in the same run.

| Dataset | Model | Acc | F1 | Params | Train (s) | ms/img | Train MB | Infer MB |
|---|---|---|---|---|---|---|---|---|
| cells | AlexNet | 94.9% | 94.7% | 2,744,424 | 52.2 | 0.064 | 166 | 114 |
| cells | VGG16 | 97.0% | 96.8% | 11,058,760 | 343.1 | 0.436 | 851 | 480 |
| cells | ResNet18 | 96.4% | 96.1% | 11,172,936 | 1026.7 | 0.975 | 2414 | 607 |
| cells | **GreenResNet** | 96.5% | 96.1% | 223,976 | 218.8 | 0.171 | 526 | 149 |
| chest | AlexNet | 83.7% | 80.3% | 2,733,570 | 17.9 | 0.060 | 163 | 112 |
| chest | VGG16 | 89.7% | 88.4% | 11,054,530 | 132.5 | 0.430 | 849 | 477 |
| chest | ResNet18 | 85.3% | 82.6% | 11,168,706 | 392.2 | 0.971 | 2409 | 605 |
| chest | **GreenResNet** | 88.5% | 86.9% | 221,858 | 83.2 | 0.186 | 525 | 147 |
| lesions | AlexNet | 75.8% | 43.1% | 2,743,399 | 29.9 | 0.065 | 167 | 114 |
| lesions | VGG16 | 70.8% | 32.3% | 11,058,247 | 201.6 | 0.442 | 851 | 479 |
| lesions | ResNet18 | 74.5% | 43.7% | 11,172,423 | 597.5 | 0.968 | 2411 | 607 |
| lesions | **GreenResNet** | 74.0% | 47.2% | 223,719 | 128.5 | 0.176 | 526 | 149 |
| orgs | AlexNet | 89.5% | 88.4% | 2,742,795 | 52.7 | 0.059 | 163 | 112 |
| orgs | VGG16 | 90.5% | 89.0% | 11,059,147 | 383.9 | 0.434 | 849 | 477 |
| orgs | ResNet18 | 91.5% | 90.3% | 11,173,323 | 1154.2 | 0.979 | 2410 | 605 |
| orgs | **GreenResNet** | 91.6% | 90.5% | 224,171 | 243.0 | 0.164 | 525 | 147 |

---

## 3. Green Initiative Analysis: Complexity vs Performance

The goal here is to show that the green model keeps the baseline's accuracy while costing a lot
less. We check accuracy against the originals from the same run, then show how much cheaper it is.

### Accuracy is preserved

Comparing the final GreenResNet to the best original in the same run:

| Dataset | Best original | GreenResNet | Difference |
|---|---|---|---|
| cells | VGG16 97.0% | 96.5% | -0.5% |
| chest | VGG16 89.7% | 88.5% | -1.2% |
| lesions | AlexNet 75.8% (F1 43.1) | 74.0% (F1 47.2) | -1.8% acc, but best F1 of all four |
| orgs | ResNet18 91.5% | 91.6% | +0.1% (green wins) |

The green model stays within about a percent of the best full model everywhere, and on orgs it is
actually the best model in the table. On lesions it gives up 1.8% accuracy but has the highest
macro F1 of any model, meaning it handles the minority classes better, which matters more than
raw accuracy on an imbalanced dataset. Accuracy is preserved.

Against ResNet18, the model it was shrunk from, the 50x smaller network matches or beats it on three of the four datasets (cells 96.5 vs 96.4, chest 88.5 vs 85.3, orgs 91.6 vs 91.5) and is within half a percent on the fourth (lesions 74.0 vs 74.5).

Why doesn't shrinking cost accuracy? Because ResNet18 is over-parameterized for these
datasets. With 64x64 images and only a few thousand training samples, its 11.2M parameters end up
fitting noise rather than learning better features on chest it reaches 97.9% validation accuracy
but only 85.3% on test. GreenResNet's 224K parameters simply cannot memorize as much, so the
smaller size acts as regularization and generalizes just as well. (This is specific to small datasets, on a large one ResNet18's extra
capacity would win.)

### Cost is drastically reduced

The whole point of the task is the saving. Comparing the final GreenResNet to ResNet18 (the
heaviest baseline):

| Metric | ResNet18 | GreenResNet | Improvement |
|---|---|---|---|
| Parameters | 11.17M | 0.22M | **~50x fewer** |
| Training time (cells) | 1026.7 s | 218.8 s | ~4.7x faster |
| Inference latency | 0.98 ms/img | 0.17 ms/img | ~5.7x faster |
| Peak training memory | 2414 MB | 526 MB | ~4.6x less |
| Peak inference memory | 607 MB | 149 MB | ~4.1x less |

So at essentially no accuracy cost against ResNet18, we cut model size by 50x, inference latency by ~5.7x,
and memory by ~4x. That is exactly the trade the board asked for. Since every model ran on the
same GPU, training time is a fair proxy for energy, so the ~4.7x speedup means a comparable cut
in energy footprint.

### The trade-offs

The trade-off is not free everywhere, and it is worth being straight about the two places where
the picture is more nuanced:

- **lesions is where accuracy dips most.** GreenResNet is 1.8% below AlexNet (74.0% vs 75.8%).
  But its macro F1 is the highest of all four models (47.2% vs 43.1%), so it spreads its errors
  more evenly across the seven classes instead of favouring the majority one. chest is no longer
  a weak spot: the green model beats ResNet18 there by 3.2% and trails only VGG16, by 1.2%.

- **AlexNet is actually the fastest to *train*.** Because AlexNet is shallow and downsamples
  aggressively, it trains in 18-53 seconds, quicker than GreenResNet. So GreenResNet is not the
  fastest model to train. But it is about **12x smaller** than AlexNet and more accurate on three
  of the four datasets (chest: 88.5% vs 83.7%, orgs: 91.6% vs 89.5%), which is what matters for a
  deployed diagnostic device.



---


# Task 3: Data-Scarcity Post-Mortem (organs dataset)

## The problem

The given `organs` dataset,  is very small:
500 training images across 11 classes (~45 per class), with a **200-image
test set**. ResNet18 has about 11 million parameters. Training that many
parameters on only 500 images causes overfitting the model memorises the
training images instead of learning general features, so it does well on the
training set but poorly on the unseen test set. This is why simple training on
`organs` alone is not enough. The target is at least 40% test accuracy.

## The strategy: transfer knowledge from `orgs`

Instead of learning from 500 images alone, we reuse features already learned on
our large existing dataset `orgs` (15,367 images). This works because `orgs` and
`organs` are the same kind of data same 1 channel, same 64×64 size, and the
same 11 classes. So the features learned on `orgs` (edges, shapes, organ
structures) apply directly to `organs`.

We split this into two scripts, each with one clear job:
- **`pretrain.py`** trains a ResNet18 on `orgs` and saves the learned weights to
  orgs_pretrained.pt. This file is the transferred knowledge.
- transfer.py loads those weights and adapts them to `organs`.

## 1. Knowledge Transfer Adaptation

We compare four training states of the same ResNet18 on the `organs` test
set. Each one differs only in what weights it starts from and *what it is
allowed to train*. Everything else is identical,
so the comparison is fair.

| Regime | Starts from | What trains | Idea |
|---|---|---|---|
| zero-shot | orgs weights | nothing | how much transfers for free |
| scratch | random weights | all layers | baseline, no transfer |
| fine-tune | orgs weights | all layers (small lr) | full transfer |
| frozen | orgs weights | new head only | safe transfer |

- **fine-tune**: load the orgs weights, then train all layers with a **small**
  learning rate (1e-4), so the good features are gently adjusted, not destroyed.
- **frozen**: load the orgs weights, freeze the whole backbone (it is *used* to
  extract features but *not* trained), and train only a fresh final Linear layer
  that maps the 512 features to the 11 organ classes.

## 2. Scarce-Data Benchmark Matrix


| Regime | Accuracy | Precision | Recall | Macro-F1 | ≥40%? |
|---|---|---|---|---|---|
| zero-shot | 49.0% | 46.7% | 44.0% | 40.8% | — |
| scratch | 53.0% | 48.4% | 46.0% | 43.8% | passed |
| fine-tune | **63.0%** | **62.8%** | **56.9%** | **54.6%** | passed |
| frozen | 60.0% | 56.5% | 53.6% | 50.9% | passed |

All trained regimes pass the 40% target. Fine-tune is best on every metric.


- **Zero-shot : borrowed knowledge, no training.** We load the orgs weights and
  immediately test on organs, without training at all. This measures how much the
  orgs model already knows about organs "for free". It is also a check: scoring
  far above random (~9% for 11 classes) proves the features transfer and that the
  labels mean the same organs in both datasets.

- **Scratch : no transfer (the baseline).** We build a fresh ResNet18 with random
  weights and train it on organs only, with no help from orgs. This is our
  control: it shows what happens without transfer learning, and it is the number
  the transfer methods must beat so we can prove transfer actually helped.

- **Fine-tune : full transfer.** We load the orgs weights, then keep training
  all layers on organs using a small learning rate (1e-4). Starting from
  good weights means the model already understands organs; the small learning rate
  lets it gently adjust those features to the new data without destroying them.
  This adapts the whole network to organs.

- **Frozen : safe transfer.** We load the orgs weights, freeze the whole
  backbone (it is used to extract features but its weights never change), and
  replace the final layer with a fresh Linear head that maps the 512 features to
  the 11 organ classes. The orgs network acts as a fixed feature-extractor, and
  only the small head learns. Because very few parameters can change, it is the
  safest option against overfitting on so little data.


### Why fine-tune uses a small learning rate (1e-4)

The learning rate was not chosen at random  we tested it. The orgs weights are
already good, so they only need gentle adjustment. To confirm this, we re-ran
fine-tune with a large learning rate (1e-3, the same as scratch):

| fine-tune setting | Accuracy | Macro-F1 |
|---|---|---|
| lr = 1e-4 (small, chosen) | **63.0%** | **54.6%** |
| lr = 1e-3 (large, test) | 52.5% | 43.5% |

With the large rate, fine-tune dropped ~10 points and fell close to the zero-shot
level. The reason is  large training steps push the
good orgs weights far from their learned values faster than 500 images can
correct them, so the model "forgets" the transferred knowledge. This experiment
is direct evidence that a small learning rate is necessary for fine-tuning.

## 3. Analysis

Impact of transfer Compared to the scratch baseline,
fine-tuning raises accuracy from 53.0% to 63.0% (+10 points) and macro-F1
from 43.8% to 54.6% (+10.8 points). The larger F1 gain matters because
macro-F1 treats all 11 classes equally; scratch's low F1 relative to its accuracy
is the sign of overfitting (good on common classes, weak on rare ones). Transfer
gives more balanced performance.

**Zero-shot confirms the approach is valid.** The orgs model scores 49.0% on
organs with no training at all, versus ~9% for random guessing over 11 classes.
This proves the features genuinely transfer, and that the 11 labels mean the same
organs in both datasets.

**Fine-tune vs frozen.** Fine-tune (63.0%) slightly beats frozen (60.0%).
Fine-tune can adjust all layers, so with a very similar domain and 500 images
being just enough, it specialises best. Frozen trains only ~5,600 head
parameters, so it can barely overfit (the safest option), but it cannot adapt the
deep features, so it is a little behind.


## 4. Recommendations

Based on our results, here is what we would suggest for using the organs dataset going forward:

- **For now (500 images):** use fine-tuning. It scored best on every metric and clearly passed the 40% target, so it is the safest choice to deploy.

- **If the data were even smaller:** use the frozen approach instead. It trains only a few thousand parameters, so there is very little chance of overfitting, the safer option when data is really limited.

- **As more organs images are collected:** move fully towards fine-tuning. With more data, training all the layers becomes safe and useful, so freezing the backbone is no longer needed.

- **To improve accuracy further:** try simple data augmentation like flipping or rotating the images to enlarge the small training set. Running the experiments with a few different seeds would also help, since the 200-image test set makes the results move around a bit.

---

# Conclusion

One principle ran through all three tasks: capacity, training strategy, and data
must be matched to each other.

In Task 1, no single model won everywhere VGG16 led on the simpler datasets
(cells, chest) and ResNet18 on the harder ones (lesions, orgs). Depth pays off only
when the task is hard enough and the data large enough (best on orgs, worst on
chest).

In Task 2, GreenResNet reached within ~0.5 - 1.8% of the best originals, and on orgs it beat
every one of them, while using ~50× fewer parameters, ~5.7× faster inference, and ~4× less
memory, showing most of that capacity was not needed.

In Task 3, with only 500 images, training from scratch overfit, so we
transferred features from the large orgs dataset. All regimes cleared the 40%
target, with fine-tuning best (63% accuracy, +10 points over scratch), and our
learning-rate experiment confirmed a small rate is essential.

Together, the assignment shows the same idea from three sides: match capacity to
the task (Task 1), remove excess capacity (Task 2), and borrow capacity when data
is scarce (Task 3).








