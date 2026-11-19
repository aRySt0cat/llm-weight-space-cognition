# MNIST weight-reading fixtures

Each file in `prompts/` contains the complete anonymous question and numerical checkpoint for one binary target-vs-rest classifier. Its filename is an opaque identifier. The experimenter-only `manifest.jsonl` associates each file with its digit and full-test metrics. Give a reader only the contents of one prompt per independent trial.

## Reproduction

From the repository root, install the pinned packages and run the following commands. The data path can name an existing copy of the standard Keras MNIST archive. A missing file is downloaded from the TensorFlow-hosted Keras datasets bucket and verified by SHA-256.

```sh
python -m pip install -r requirements.txt
python prepare.py prepare --data /tmp/weight-space/mnist.npz --jobs 3
python prepare.py verify --data /tmp/weight-space/mnist.npz
```

The published fixtures were generated with CPython 3.12.14 on Linux x86-64, using OpenBLAS 0.3.30 with one thread per fit. Floating-point training may vary across numerical libraries and CPU architectures. The published ASCII checkpoints can be read independently of the training backend.

## Fixed recipe

- The standard training split contains 60,000 images. The standard test split contains 10,000. The archive SHA-256 is `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.
- Every input is average-pooled over non-overlapping 2 by 2 blocks, flattened in row-major order and divided by 255. Ink increases toward 1. Inputs use NumPy `float32`.
- Each digit uses 5,000 positives and 5,000 negatives sampled without replacement from the training split. Negatives are sampled uniformly from all examples of the other digits. The same sampled rows and initial sample ordering are used for all five architectures for that digit.
- The sampling generator is NumPy `default_rng(SeedSequence([20261119, 0, digit]))`. Each estimator seed is the first 32-bit output of `SeedSequence([20261119, 1, zero_based_tier_index, digit])`.
- The linear tier uses scikit-learn logistic regression with L2 regularization, `C=1`, L-BFGS, a tolerance of `1e-5`, and a 500-iteration limit.
- The four hidden-layer tiers use scikit-learn's `MLPClassifier` with ReLU, Adam, a learning rate of 0.002, L2 coefficient 0.0001, batch size 128, and 25 shuffled epochs. Early stopping is disabled. The remaining optimizer settings use scikit-learn 1.8.0 defaults. All digit and tier combinations use the same predeclared recipe without tuning on test results.
- IDs are drawn as 64-bit opaque values with an independent `random.Random(914658237)` generator and shuffled before assignment. The study does not expose IDs, keys, labels or metrics in prompts.
- The parameter strings use Python's `.5g` formatting and omit the zero before a decimal point in values with magnitude below one. Each input-facing unit's incoming weights form fourteen fourteen-value scan lines. Later units have one row of incoming weights. The final output is an affine logit. Every bias is included in the parameter count.
- The script parses the checkpoint as `float64`, evaluates every test example, and measures balanced accuracy (mean sensitivity and specificity using a logit greater than zero) and AUROC (raw logits). There is no separately stored full-precision checkpoint dependency.

| Tier | Widths | Parameters |
| --- | --- | ---: |
| `linear` | 196, 1 | 197 |
| `h4` | 196, 4, 1 | 793 |
| `h32` | 196, 32, 1 | 6,337 |
| `h32x32` | 196, 32, 32, 1 | 7,393 |
| `h128x64` | 196, 128, 64, 1 | 33,537 |

Each manifest row contains `id` (filename stem), `tier`, `digit`, `params`, `bytes` (complete ASCII prompt including final newline), `sha256` (same bytes), `balanced_accuracy`, and `auc`. The verification command checks the exact fixture set, file sizes, SHA-256 digests, parameter counts, canonical serialization, architectures, and recomputed metrics.
