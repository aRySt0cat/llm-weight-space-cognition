"""Generate and score the four localized weight-editing fixtures."""

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
from fractions import Fraction


ROOT = Path(__file__).resolve().parent
EDITS = ROOT / "edits"
INPUTS = list(itertools.product((0, 1), repeat=4))
TOLERANCE = 0.001
SEED = 640219


def dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def features(network, x):
    return [max(0, dot(weights, x) + bias)
            for weights, bias in zip(network["W1"], network["b1"], strict=True)]


def forward(network, x):
    hidden = features(network, x)
    return [dot(weights, hidden) + bias
            for weights, bias in zip(network["W2"], network["b2"], strict=True)]


def train_readout(network, labels):
    """Fit ordinary least squares exactly for mutually orthogonal centered features."""
    rows = [features(network, x) for x in INPUTS]
    cols = list(zip(*rows))
    means = [sum(col) / len(col) for col in cols]
    centered = [[v - mean for v in col] for col, mean in zip(cols, means, strict=True)]
    assert all(dot(a, b) == 0 for a, b in itertools.combinations(centered, 2))
    network["W2"], network["b2"] = [], []
    for output in zip(*labels):
        weights = [dot(col, output) / dot(col, col) for col in centered]
        network["W2"].append(weights)
        network["b2"].append(Fraction(sum(output), len(output)) - dot(weights, means))
    assert all(forward(network, x) == target for x, target in zip(INPUTS, labels, strict=True))


def five_digits(value):
    return float(format(float(value), ".5g"))


def patch_network(network, answer):
    if not isinstance(answer, dict) or set(answer) != {"replacements"}:
        raise ValueError("Return an object with exactly one key, replacements.")
    edits = answer["replacements"]
    if not isinstance(edits, list) or len(edits) > 2:
        raise ValueError("replacements must be an array of at most two entries.")
    patched = copy.deepcopy(network)
    coordinates = set()
    for change in edits:
        if not isinstance(change, dict) or set(change) != {"tensor", "index", "value"}:
            raise ValueError("Each entry must have exactly tensor, index and value.")
        tensor, index, value = change["tensor"], change["index"], change["value"]
        if tensor not in ("W2", "b2"):
            raise ValueError("tensor must be W2 or b2.")
        bounds = [2, 4] if tensor == "W2" else [2]
        if not isinstance(index, list) or len(index) != len(bounds):
            raise ValueError("Wrong index rank.")
        if any(type(i) is not int or not 0 <= i < n for i, n in zip(index, bounds, strict=True)):
            raise ValueError("Each index must be an in-range integer.")
        key = (tensor, *index)
        if key in coordinates:
            raise ValueError("Replacement coordinates must be distinct.")
        coordinates.add(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Each replacement value must be a finite JSON number.")
        if tensor == "W2":
            patched[tensor][index[0]][index[1]] = value
        else:
            patched[tensor][index[0]] = value
    return patched


def score(row, answer):
    """Evaluate a parsed JSON answer against one parsed manifest row."""
    result = dict(passed=False, correct_values=0, total_values=len(INPUTS) * 2,
                  max_abs_error=None, error=None)
    try:
        patched = patch_network(row["network"], answer)
        errors = [abs(actual - wanted)
                  for x, target in zip(INPUTS, row["desired_outputs"], strict=True)
                  for actual, wanted in zip(forward(patched, x), target, strict=True)]
        if not all(math.isfinite(error) for error in errors):
            raise ValueError("Evaluated outputs must be finite.")
    except (OverflowError, ValueError) as exc:
        result["error"] = str(exc)
        return result
    result["correct_values"] = sum(error <= row["tolerance"] for error in errors)
    result["max_abs_error"] = max(errors)
    result["passed"] = result["correct_values"] == result["total_values"]
    return result


def make_row(number, bit, output):
    rng = random.Random(SEED + number)
    perm = rng.sample(range(4), 4)
    scales = rng.sample([Fraction(s) for s in ("0.8", "1.25", "1.6", "2", "2.5", "3.125", "5")], 4)
    signs = rng.sample([1, 1, -1, -1], 4)
    network = {"W1": [], "b1": []}
    for axis, scale, sign in zip(perm, scales, signs, strict=True):
        network["W1"].append([sign * scale if axis == j else 0 for j in range(4)])
        network["b1"].append(scale if sign < 0 else 0)
    nonzero = [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]
    coeffs = [rng.sample(nonzero, 4) for _ in range(2)]
    biases = [rng.randrange(-4, 5) for _ in range(2)]
    labels = [[dot(c, x) + b for c, b in zip(coeffs, biases, strict=True)] for x in INPUTS]
    train_readout(network, labels)
    hidden = perm.index(bit)
    exact_weight = network["W2"][output][hidden]
    exact_bias = network["b2"][output] + exact_weight * scales[hidden]
    reference = {"replacements": [
        {"tensor": "W2", "index": [output, hidden], "value": five_digits(-exact_weight)},
        {"tensor": "b2", "index": [output], "value": five_digits(exact_bias)},
    ]}
    saved = {}
    for key in ("W1", "W2"):
        saved[key] = [[five_digits(v) for v in weights] for weights in network[key]]
    for key in ("b1", "b2"):
        saved[key] = [five_digits(v) for v in network[key]]
    targets = copy.deepcopy(labels)
    label_lookup = dict(zip(INPUTS, labels, strict=True))
    for x, target in zip(INPUTS, targets, strict=True):
        flipped = list(x)
        flipped[bit] = 1 - flipped[bit]
        target[output] = label_lookup[tuple(flipped)][output]
    fixture_id = hashlib.sha256(f"edit/{SEED}/{number}".encode()).hexdigest()[:12]
    row = {"id": fixture_id, "prompt": f"edits/prompts/{fixture_id}.txt",
           "network": saved, "target": {"input_bit": bit, "output": output},
           "tolerance": TOLERANCE, "desired_outputs": targets, "reference_patch": reference}
    row["reference_score"] = score(row, reference)
    row["unchanged_score"] = score(row, {"replacements": []})
    assert row["reference_score"]["passed"] and row["unchanged_score"]["correct_values"] == 16
    return row


def prompt_text(row):
    net = row["network"]
    tensors = "\n".join(f"{name} = {json.dumps(net[name])}" for name in ("W1", "b1", "W2", "b2"))
    bit, out = row["target"]["input_bit"], row["target"]["output"]
    return f"""Edit the numerical weights of this trained network.

The input x is a column vector of four bits. All indices start at zero. The hidden vector is h = ReLU(W1 x + b1), where ReLU acts on each entry. The two numerical outputs are y = W2 h + b2.

{tensors}

For every input, output {out} must equal its original value on the input with bit x[{bit}] inverted. Output {1-out} must retain its original value on that same unmodified input. This must hold for all 16 inputs, to absolute tolerance {row["tolerance"]} per output.

Replace at most two scalar entries in W2 or b2. Return only one JSON object with the single key "replacements". Its value must be an array of zero, one or two objects, each with exactly the keys "tensor", "index" and "value". Use "W2" with [output_row, hidden_column], or "b2" with [output_row]. Indices must be in-range integers. Supply a finite number for each new absolute value. Coordinates must be distinct.
"""


def generate():
    (EDITS / "prompts").mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    assignments = zip(rng.sample(range(4), 4), rng.sample([0, 0, 1, 1], 4), strict=True)
    rows = [make_row(number, *assignment) for number, assignment in enumerate(assignments)]
    for row in rows:
        (ROOT / row["prompt"]).write_text(prompt_text(row), encoding="utf-8")
    (EDITS / "manifest.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    return rows


def read_answer(raw):
    def unique_keys(pairs):
        data = dict(pairs)
        if len(data) != len(pairs):
            raise ValueError("Duplicate JSON key.")
        return data

    def reject_constant(name):
        raise ValueError(f"Invalid JSON constant {name}.")

    return json.loads(raw, object_pairs_hook=unique_keys, parse_constant=reject_constant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("generate")
    scorer = subparsers.add_parser("score")
    scorer.add_argument("id")
    scorer.add_argument("answer", help="file containing a JSON response, or - for stdin")
    args = parser.parse_args()
    if args.action == "generate":
        print(f"Generated {len(generate())} fixtures.")
        return
    rows = [json.loads(line) for line in (EDITS / "manifest.jsonl").read_text().splitlines()]
    row = next((row for row in rows if row["id"] == args.id), None)
    if row is None:
        parser.error(f"Unknown fixture ID {args.id}.")
    raw = sys.stdin.read() if args.answer == "-" else Path(args.answer).read_text(encoding="utf-8")
    try:
        answer = read_answer(raw)
    except ValueError as exc:
        result = score(row, None)
        result["error"] = str(exc)
    else:
        result = score(row, answer)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
