# Incident Audit Log

This log itemizes every bug, corruption, and anti-pattern found in the recovered codebase (`data.py`, `models.py`, `train.py`, `fit.py`) and the structural correction applied to each. The recovered file containing the `Trainer` class is named `fit.py` in the provided template; the brief refers to it generically as `trainer.py`, we kept the template's name.

Entries are listed in the **order they were discovered and fixed**. The workflow was iterative, run the code, hit a crash or a bad metric, find the cause, fix it, commit, run again, so each row is one step of that debugging sequence and maps to one commit.

For every entry: **(i)** file, **(ii)** how the problem manifests, **(iii)** the mathematical/logical root cause, **(iv)** the structural correction, **(v)** the git commit hash containing the fix.

| # | File | Manifestation | Root Cause | Correction | Commit |
|---|------|---------------|------------|------------|--------|
| 1 | `train.py` | `FileNotFoundError: config.json` on the first line; the pipeline never starts. | Every hyperparameter is read from `config.json`, which was deleted in the incident and absent from the recovered code. | Recreated `config.json` with all pipeline settings: `DATA`, `DATA_PATH`, `BATCH_SIZE`, `MODEL`, `CHANNELS`, `NUM_CLASSES`, `LEARNING_RATE`, `EPOCHS`, `DROP_RATE`, `ACTIVATION`. | `2eecab6` |
| 2 | `data.py` | `FileNotFoundError: data\cells_data.pt`; crash while loading any dataset. | Path was built as `f"{data}_data.pt"` → `cells_data.pt`, but the restored files are named `cells.pt`. | Changed the path template to `f"{data}.pt"`. | `f088fdc` |
| 3 | `models.py` (`ResNet18.forward`) | `TypeError: cross_entropy_loss(): argument 'input' must be Tensor, not NoneType`. | `forward()` computed `self.classifier(out)` on the last line but never returned it, so the model output was `None`. | Added `return self.classifier(out)`. | `f25675e` |
| 4 | `data.py` | `RuntimeError: 0D or 1D target tensor expected, multi-target not supported`. | Labels were stored with shape `[N, 1]`; `CrossEntropyLoss` expects 1-D class indices `[N]`. | Applied `.squeeze(1)` to train / val / test labels (dtype was already `int64`). | `3d2a7a1` |
| 5 | `fit.py` (`train_one_epoch`) | Training loss *grows* each epoch (3.0 → 6.1 → 13 → 20 → 29); accuracy stuck ~23%. | `optimizer.zero_grad()` was missing, so gradients from every batch accumulated instead of resetting. | Added `self.optimizer.zero_grad()` at the start of every batch, before the backward pass. | `c5762a5` |
| 6 | `data.py` | No error, validation accuracy suspiciously tracks/exceeds training (data leak). | `train_data` used the **whole** array while `val_data` was a slice *of that same array*, so validation samples were also in training. | Sliced train to `[:val_start]` and val to `[val_start:]` so the sets are disjoint. | `77cdbda` |
| 7 | `models.py` | Log prints `Using activation function: Identity()`; accuracy capped well below target. | `activation_str` defaulted to `"Identity"`, giving no non-linearity, a stack of linear layers collapses into a single linear map, so a deep net has no more capacity than one layer. | Set / read the activation as `ReLU`.| `42781cd`  |
| 8 | `models.py` (`AlexNet`) | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x3072 and 2048x1024)`. | The first `Linear` assumed a fixed flatten size (`2048`) that only holds for one input resolution, with no pooling to normalise spatial size. | Added `nn.AdaptiveAvgPool2d((1, 1))` before the flatten and set the classifier input to the channel count ( `192` AlexNet). | `58d7e11` |
| 9| `models.py` (`AlexNet`) | Crashes on 1-channel datasets (chest, orgs, organs); wrong output size whenever class count ≠ 11. | The first conv hardcoded `Conv2d(3, …)` and the final layer hardcoded `Linear(1024, 11)`, ignoring `in_channels` / `num_classes`. | Parameterized the constructor: first conv uses `in_channels`, final linear uses `num_classes` (confirmed by inspection: cells/lesions = 3ch, chest/orgs/organs = 1ch). | `58d7e11` |
| 10|`train.py` / `models.py` | Loss frozen ~2.01, accuracy stuck ~19% every epoch. | `drop_rate` was hardcoded to `0.99`, zeroing ~99% of activations each forward pass, almost no signal survives, so the network cannot learn. | Read dropout from config (`config["DROP_RATE"]`, set to `0.5`). | `b2b1a4c` |
| 11|`models.py` (`VGGBlock`) | `RuntimeError: Given groups=1, weight of size [64,3,3,3], expected input to have 3 channels, but got 64`. | `current_in_channels` was never updated inside the conv loop, so the 2nd conv received the 1st conv's 64-channel output while still expecting 3. | Added `current_in_channels = out_channels` on each loop iteration. | `256641e` |
| 12| `models.py` (`VGG16`) | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x3072 and 2048x1024)`. |The first `Linear` assumed a fixed flatten size (`2048`) that only holds for one input resolution, with no pooling to normalise spatial size | Added `nn.AdaptiveAvgPool2d((1, 1))` before the flatten and set the classifier input to the channel count (`512` VGG16). | `32b3bcb` |
| 13 | `train.py` | Different accuracy every run on the same setup (e.g. 47% vs 85%). | No fixed random seed, so weight initialisation and data shuffling differ each run. | Added `torch.manual_seed(42)` | `4bbffe5` |
| 14 | `data.py` | No error; a class-sorted file can place whole classes only in train or only in val. | The split sliced the array in order, so class-ordered data produces a biased validation distribution. | Shuffled with a fixed-seed `torch.randperm` **before** slicing. | `b1e7da8` |
| 15 | `fit.py` (`train_one_epoch`) | None, identical results before and after (latent anti-pattern). | `correct, sum = 0, 0` rebinds the name `sum` to an int, shadowing Python's built-in inside the function. | Renamed the counter to `total` (matching `evaluate`, which already used `total`). | `ffa249a` |



# Training Results Log

Some of the bugs did not crash the code. We only found them by reading the training output and noticing the numbers looked wrong , loss going up, validation beating training, loss frozen.


---

## Bug 5 : Missing `optimizer.zero_grad()` (fit.py)

**What we saw:** training loss went *up* every epoch instead of down, and accuracy stayed stuck around 23%. Gradients from every batch were piling up instead of resetting.

Before the fix:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:-----:|:----------:|:---------:|:--------:|:-------:|
| 1 | 3.0664 | 33.33% | 7.2155 | 28.90% |
| 2 | 6.1449 | 22.76% | 22.6061 | 16.61% |
| 3 | 13.2078 | 26.32% | 11.2056 | 25.24% |
| 4 | 20.0932 | 24.77% | 25.4972 | 21.21% |
| 5 | 29.3557 | 23.15% | 36.6196 | 32.04% |

After adding `zero_grad()`, the loss started dropping normally instead of exploding.

---

## Bug 6 : Data leak in the train/val split (data.py)

**What we saw:** validation accuracy kept matching or even beating training accuracy (look at epochs 8–10).

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:-----:|:----------:|:---------:|:--------:|:-------:|
| 1 | 1.1744 | 57.69% | 1.1800 | 55.89% |
| 2 | 0.9656 | 64.37% | 1.0358 | 59.84% |
...
| 8 | 0.6253 | 78.34% | 0.5463 | **81.35%** |
| 9 | 0.5952 | 79.51% | 0.5695 | **80.98%** |
| 10 | 0.5779 | 80.02% | 0.5587 | **80.98%** |
....

After making the train and val sets disjoint, validation stopped tracking training so closely.

---

## Bug 7 : No activation function / Identity (models.py)

**What we saw:** the log printed `Using activation function: Identity()`, so the network had no non-linearity. After switching to ReLU, accuracy jumped up.

After the fix (ReLU):

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:-----:|:----------:|:---------:|:--------:|:-------:|
| 1 | 0.4192 | 85.22% | 0.9259 | 72.71% |
| 2 | 0.2098 | 92.73% | 3.5013 | 49.16% |
| 3 | 0.1736 | 94.08% | 0.5515 | 81.49% |
| 4 | 0.1387 | 95.19% | 0.8909 | 80.47% |
| 5 | 0.1524 | 94.80% | 0.3276 | 90.20% |

---

## Bug 8 : Missing adaptive pooling (AlexNet / VGG16)

**What we saw:** first a crash  `mat1 and mat2 shapes cannot be multiplied`. After adding `AdaptiveAvgPool2d`, the crash went away and training ran, but the loss got stuck at about 2.0 and accuracy at ~19%. That frozen result is what pointed us to the next bug.

Ran, but frozen:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:-----:|:----------:|:---------:|:--------:|:-------:|
| 1 | 2.0492 | 18.22% | 2.0188 | 19.17% |
| 2 | 2.0203 | 18.90% | 2.0172 | 19.17% |
| 3 | 2.0152 | 18.29% | 2.0189 | 19.17% |
| 4 | 2.0106 | 18.64% | 2.0129 | 19.17% |
| 5 | 2.0086 | 19.34% | 2.0143 | 19.17% |

---

## Bug 10 : Dropout set too high (0.99)

**What we saw:** loss frozen at ~2.0 and accuracy stuck at ~19% (the table above). Dropping 99% of the activations left almost no signal for the network to learn from. After lowering dropout to 0.5, the loss dropped.

After the fix (dropout 0.5):

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:-----:|:----------:|:---------:|:--------:|:-------:|
| 1 | 1.0980 | 56.72% | 3.7588 | 23.56% |
| 2 | 0.5293 | 79.36% | 2.2868 | 39.28% |
| 3 | 0.3722 | 87.00% | 0.6270 | 78.71% |
| 4 | 0.3037 | 90.17% | 0.6062 | 76.88% |
| 5 | 0.2306 | 92.51% | 0.7071 | 80.61% |

---


