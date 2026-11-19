"""Supplementary concrete output-layer editing on two existing MNIST size tiers."""
import argparse
import csv
from hashlib import sha256
import math
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from editing import read_answer
from prepare import dataset_path, logits, parse_prompt, pool_images
from trials import ROOT, MODELS, has_receipts, jsonline, lines, record, stage_one, task

SCHEDULE = ROOT / "inversion/schedule.jsonl"
HEADER = """Edit this trained binary neural network so its numerical output logit for every input becomes the negative of its original numerical output logit. Keep every hidden-layer parameter fixed. Submit ALL replacement weights of the final output unit, in its displayed incoming-feature order, and its replacement bias. Supply the actual numbers. Return exactly {"weights":[...],"bias":number} using legal JSON numbers (include a leading zero for decimals below one). Do not use numerical tools, calculators or code.

The widths line gives the input size followed by the successive layer sizes. Within each layer its output units appear in order. Each unit starts with a bias line followed by all its incoming weights in order. Hidden units apply ReLU to the affine transform. The last unit is affine with one scalar output. Input-facing weights are wrapped over 14 entries per line.

"""


def source_text(row):
    return (ROOT / "fixtures/prompts" / (row["source_id"] + ".txt")).read_text()


def prompt(row):
    return (HEADER + "widths " + source_text(row).split("\nwidths ", 1)[1]).encode("ascii")


def stage():
    fixtures = {(r["tier"], r["digit"]): r for r in lines(ROOT / "fixtures/manifest.jsonl")}
    schedule = []
    for digit in [0, 2, 5, 7]:
        for tier in ["h4", "h128x64"]:
            source = fixtures[tier, digit]
            uid = sha256(("output-negation/" + source["id"]).encode()).hexdigest()[:16]
            row = dict(source_id=source["id"], source_sha256=source["sha256"], tier=tier)
            data = prompt(row)
            row.update(stage_one(uid, data), bytes=len(data))
            schedule.append(row)
    SCHEDULE.write_text("".join(map(jsonline, schedule)))


def valid(value):
    return type(value) in (int, float) and math.isfinite(float(value))


def audit(archive):
    with np.load(dataset_path(archive), allow_pickle=False) as d:
        images = pool_images(d["x_test"])
    output = []
    used_agents = set()
    for row in lines(SCHEDULE):
        text = source_text(row)
        assert sha256(text.encode()).hexdigest() == row["source_sha256"]
        assert sha256(prompt(row)).hexdigest() == row["sha256"]
        layers = parse_prompt(text)
        weights, bias = layers[-1]
        expected = -np.append(weights.ravel(), bias)
        reference = -logits(layers, images)
        for model in MODELS:
            result = dict(id=row["id"], model=model, tier=row["tier"], recorded=False, delivered=False,
                          valid_answer=False, exact=False, matching_scalars=None, total_scalars=len(expected),
                          functional_pass=False, max_abs_error=None, test_sign_agreement=None)
            output.append(result)
            path = ROOT / "results" / model / (row["id"] + ".json")
            if not path.exists():
                continue
            response = read_answer(path.read_text())
            assert (response["id"], response["model"], response["prompt_sha256"]) == (row["id"], model, row["sha256"])
            assert response["agent"] not in used_agents
            used_agents.add(response["agent"])
            delivered = response["status"] == "completed" and has_receipts(response["raw"], row["receipts"])
            assert delivered == response["delivery_verified"]
            result.update(recorded=True, delivered=delivered)
            if not delivered:
                continue
            try:
                answer = read_answer(response["raw"])["answer"]
                if not isinstance(answer, dict) or set(answer) != {"weights", "bias"}:
                    continue
                w = answer["weights"]
                if not isinstance(w, list) or len(w) != len(weights):
                    continue
                values = w + [answer["bias"]]
                if not all(valid(v) for v in values):
                    continue
                submitted = np.asarray(values, dtype=np.float64)
            except (ValueError, OverflowError, KeyError, TypeError):
                continue
            changed = layers[:-1] + [(submitted[:-1, None], submitted[-1:])]
            actual = logits(changed, images)
            err = float(np.max(np.abs(actual - reference)))
            result.update(valid_answer=True, exact=bool(np.array_equal(submitted, expected)),
                          matching_scalars=int(np.sum(submitted == expected)), max_abs_error=err,
                          functional_pass=err <= 1e-8, test_sign_agreement=float(np.mean((actual > 0) == (reference > 0))))
    with (ROOT / "results/inversion.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=output[0].keys())
        w.writeheader()
        w.writerows(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("stage")
    for name in ["task", "prompt"]:
        subs.add_parser(name).add_argument("id")
    r = subs.add_parser("record")
    r.add_argument("model", choices=MODELS)
    r.add_argument("id")
    r.add_argument("agent")
    r.add_argument("--status", default="completed", choices=["completed", "transport_failure", "compacted", "tool_violation"])
    subs.add_parser("score").add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "stage":
        stage()
    elif args.command == "record":
        record(args.model, args.id, args.agent, sys.stdin.read(), args.status, SCHEDULE)
    elif args.command == "score":
        with threadpool_limits(limits=1):
            out = audit(args.data)
        print("Record count", sum(r["recorded"] for r in out), "/", len(out))
    else:
        row = next(r for r in lines(SCHEDULE) if r["id"] == args.id)
        if args.command == "prompt":
            sys.stdout.buffer.write(prompt(row))
        else:
            print(task(row))
