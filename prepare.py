"""Prepare or independently verify the blinded MNIST weight-reading fixtures.

The fixed training and blinding recipe below is independent of test-set results.
Only the entire standard MNIST test split is used for the reported metrics.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import tempfile
import urllib.request
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
TIERS = {
    "linear": (),
    "h4": (4,),
    "h32": (32,),
    "h32x32": (32, 32),
    "h128x64": (128, 64),
}
TRAIN_SEED = 20261119
BLIND_SEED = 914658237
MAX_BYTES = 300_000
N_PER_CLASS = 5_000
MLP_EPOCHS = 25
PROMPT = """This binary classifier was trained to recognize one MNIST digit against all other digits. Infer the target digit (0 through 9) from the parameters below. Return only a JSON object of the form {"digit": n}. Use no tools, code execution, calculators, or external aids.

The input is a 14 by 14 grayscale image made by averaging non-overlapping 2 by 2 blocks in the original MNIST image. Ink is 1 and background is 0. Flatten it in row-major order, left to right and top to bottom. The widths line gives the input and successive layer sizes. Each layer lists its output units in order. For each unit the bias line is followed by the incoming weights in input order. Input-facing weights use one line per image row. Each hidden layer applies ReLU after its affine transform. The final unit produces a logit; a positive logit favors the target digit.
"""


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dataset_path(path):
    """Use a local copy, or download and verify the pinned public archive."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            urllib.request.urlretrieve(MNIST_URL, temporary)
            if checksum(temporary) != MNIST_SHA256:
                raise ValueError("Downloaded MNIST archive failed the SHA-256 check")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if checksum(path) != MNIST_SHA256:
        raise ValueError(f"Unexpected MNIST archive SHA-256 for {path}")
    return path


def pool_images(images):
    """Map standard 28x28 uint8 images to [0, 1] float32 row-major vectors."""
    return images.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4), dtype=np.float32).reshape(-1, 196) / np.float32(255)


def balanced_indices(labels, digit):
    """Use identical examples across all five architectures for a given digit."""
    rng = np.random.default_rng(np.random.SeedSequence([TRAIN_SEED, 0, digit]))
    positives = rng.choice(np.flatnonzero(labels == digit), N_PER_CLASS, replace=False)
    negatives = rng.choice(np.flatnonzero(labels != digit), N_PER_CLASS, replace=False)
    return rng.permutation(np.concatenate((positives, negatives)))


def train_model(x, y, tier, digit):
    idx = balanced_indices(y, digit)
    seed = int(np.random.SeedSequence([TRAIN_SEED, 1, list(TIERS).index(tier), digit]).generate_state(1)[0])
    if tier == "linear":
        model = LogisticRegression(C=1, solver="lbfgs", max_iter=500, tol=1e-5, random_state=seed)
        model.fit(x[idx], y[idx] == digit)
        return [(model.coef_.T, model.intercept_)]
    model = MLPClassifier(
        hidden_layer_sizes=TIERS[tier], activation="relu", solver="adam",
        alpha=1e-4, batch_size=128, learning_rate_init=0.002,
        max_iter=MLP_EPOCHS, early_stopping=False, shuffle=True,
        tol=0.0, n_iter_no_change=MLP_EPOCHS + 1, random_state=seed,
    )
    model.fit(x[idx], y[idx] == digit)
    return list(zip(model.coefs_, model.intercepts_))


def format_weight(value):
    token = format(float(value), ".5g")
    if token.startswith("-0."):
        return "-" + token[2:]
    if token.startswith("0."):
        return token[1:]
    return token


def serialize_prompt(layers):
    """Serialize (input x output weights, output biases) into the model prompt."""
    widths = [layers[0][0].shape[0], *(len(biases) for _, biases in layers)]
    lines = [PROMPT, "widths " + " ".join(map(str, widths))]
    for i, (weights, biases) in enumerate(layers):
        if weights.shape != (widths[i], widths[i + 1]):
            raise ValueError("Layer dimensions do not match")
        lines.append(f"layer {i + 1}")
        for incoming, bias in zip(weights.T, biases):
            lines.append("bias " + format_weight(bias))
            row_width = 14 if i == 0 else len(incoming)
            for start in range(0, len(incoming), row_width):
                lines.append(" ".join(map(format_weight, incoming[start:start + row_width])))
    return "\n".join(lines) + "\n"


def parse_prompt(text):
    """Decode the attached checkpoint while checking its declared dimensions."""
    lines = iter(text.splitlines())
    width_line = next((line for line in lines if line.startswith("widths ")), None)
    if width_line is None:
        raise ValueError("Missing layer widths")
    widths = list(map(int, width_line.split()[1:]))
    if len(widths) < 2 or widths[0] != 196 or widths[-1] != 1 or min(widths) < 1:
        raise ValueError("Invalid architecture")
    layers = []
    for i, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
        if next(lines, None) != f"layer {i + 1}":
            raise ValueError("Missing layer header")
        weights = np.empty((inputs, outputs), dtype=np.float64)
        biases = np.empty(outputs, dtype=np.float64)
        row_width = 14 if i == 0 else inputs
        for unit in range(outputs):
            bias_line = next(lines, "").split()
            if len(bias_line) != 2 or bias_line[0] != "bias":
                raise ValueError("Invalid bias line")
            biases[unit] = float(bias_line[1])
            for start in range(0, inputs, row_width):
                tokens = next(lines, "").split()
                if len(tokens) != row_width:
                    raise ValueError("Invalid weight row")
                weights[start:start + row_width, unit] = tokens
        if not np.isfinite(weights).all() or not np.isfinite(biases).all():
            raise ValueError("Non-finite parameter")
        layers.append((weights, biases))
    if any(line.strip() for line in lines):
        raise ValueError("Unexpected trailing parameter data")
    return layers


def logits(layers, x):
    """Compute raw binary scores from parsed row-vector affine transforms."""
    hidden = x
    for weights, biases in layers[:-1]:
        hidden = np.maximum(hidden @ weights + biases, 0)
    weights, biases = layers[-1]
    return (hidden @ weights + biases).ravel()


def evaluate(layers, x_test, y_test, digit):
    scores = logits(layers, x_test)
    labels = y_test == digit
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, scores > 0)),
        "auc": float(roc_auc_score(labels, scores)),
    }


def assignments():
    """Assign opaque 64-bit IDs using a PRNG independent of the training seeds."""
    rng = random.Random(BLIND_SEED)
    pairs = [(tier, digit) for tier in TIERS for digit in range(10)]
    ids = set()
    while len(ids) < len(pairs):
        ids.add(f"{rng.getrandbits(64):016x}")
    shuffled = sorted(ids)
    rng.shuffle(shuffled)
    return [(fixture_id, tier, digit) for fixture_id, (tier, digit) in zip(shuffled, pairs)]


def describe(text, fixture_id, tier, digit, x_test, y_test):
    raw = text.encode("ascii")
    if len(raw) > MAX_BYTES:
        raise ValueError(f"{fixture_id} ({tier}) is {len(raw)} bytes, over the {MAX_BYTES} limit")
    layers = parse_prompt(text)
    if tuple(w.shape[1] for w, _ in layers[:-1]) != TIERS[tier]:
        raise ValueError(f"{fixture_id} architecture disagrees with tier")
    return {
        "id": fixture_id, "tier": tier, "digit": digit,
        "params": sum(w.size + b.size for w, b in layers), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        **evaluate(layers, x_test, y_test, digit),
    }


def prepare(output, archive, jobs):
    x_train, y_train = pool_images(archive["x_train"]), archive["y_train"]
    x_test, y_test = pool_images(archive["x_test"]), archive["y_test"]
    prompts = output / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)

    def build(assignment):
        fixture_id, tier, digit = assignment
        layers = train_model(x_train, y_train, tier, digit)
        text = serialize_prompt(layers)
        record = describe(text, fixture_id, tier, digit, x_test, y_test)
        return record, text

    # The intended epoch limit produces this standard MLP training warning.
    with warnings.catch_warnings(), threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=jobs) as workers:
        warnings.filterwarnings("ignore", message="Stochastic Optimizer: Maximum iterations", category=ConvergenceWarning)
        with tempfile.NamedTemporaryFile(mode="w", dir=output, encoding="ascii", delete=False) as manifest:
            tmp_path = Path(manifest.name)
            try:
                for record, text in workers.map(build, assignments()):
                    (prompts / (record["id"] + ".txt")).write_bytes(text.encode("ascii"))
                    manifest.write(json.dumps(record, separators=(",", ":")) + "\n")
                    print(f'{record["id"]} {record["tier"]} {record["digit"]} {record["bytes"]} bytes', flush=True)
                manifest.flush()
                tmp_path.replace(output / "manifest.jsonl")
            finally:
                tmp_path.unlink(missing_ok=True)


def verify(output, archive):
    x_test, y_test = pool_images(archive["x_test"]), archive["y_test"]
    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    expected = {(fixture_id, tier, digit) for fixture_id, tier, digit in assignments()}
    actual = {(r["id"], r["tier"], r["digit"]) for r in records}
    if len(records) != len(expected) or actual != expected:
        raise ValueError("Manifest does not cover the predeclared assignments exactly once")
    prompt_paths = {p.stem for p in (output / "prompts").glob("*.txt")}
    if prompt_paths != {r["id"] for r in records}:
        raise ValueError("Prompt file set disagrees with manifest")
    with threadpool_limits(limits=1):
        for record in records:
            path = output / "prompts" / (record["id"] + ".txt")
            text = path.read_bytes().decode("ascii")
            current = describe(text, record["id"], record["tier"], record["digit"], x_test, y_test)
            if current != record or serialize_prompt(parse_prompt(text)) != text:
                raise ValueError(f"Verification failed for {path}")
    print(f"Verified {len(records)} prompts, canonical weights, hashes, and full-test metrics.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="prepare", choices=("prepare", "verify"))
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/weight-space/mnist.npz",
                        help="Local MNIST NPZ path (downloads there if missing)")
    parser.add_argument("--output", type=Path, default=ROOT / "fixtures")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent fits; each fit uses one BLAS thread")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    with np.load(dataset_path(args.data), allow_pickle=False) as archive:
        if args.mode == "verify":
            verify(args.output, archive)
        else:
            prepare(args.output, archive, args.jobs)


if __name__ == "__main__":
    main()
