"""Supplementary external controls on the *same* decimal checkpoints given to raters."""
import argparse
import csv
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from prepare import dataset_path, logits, parse_prompt, pool_images
from trials import ROOT, jsonline, lines


def count_tokens():
    import tiktoken
    encoder = tiktoken.get_encoding("o200k_base")
    records = []
    for row in lines(ROOT / "schedule.jsonl"):
        folder = "edits" if row["task"] == "edit" else "fixtures"
        data = (ROOT / folder / "prompts" / (row["id"] + ".txt")).read_bytes()
        records.append(dict(id=row["id"], bytes=len(data), o200k_base=len(encoder.encode(data.decode("ascii")))))
    (ROOT / "results/sizes.jsonl").write_text("".join(map(jsonline, records)))


def activation_probe(archive):
    with np.load(dataset_path(archive), allow_pickle=False) as d:
        labels = d["y_test"]
        # The same 50 canonical test images are used for every network.
        indices = np.stack([np.flatnonzero(labels == digit)[:5] for digit in range(10)])
        x = pool_images(d["x_test"][indices.ravel()])
    output = []
    with threadpool_limits(limits=1):
        for row in lines(ROOT / "fixtures/manifest.jsonl"):
            checkpoint = (ROOT / "fixtures/prompts" / (row["id"] + ".txt")).read_text()
            scores = logits(parse_prompt(checkpoint), x).reshape(10, 5)
            # argmax tie break is the smaller digit; consult target only *after* prediction.
            guess = int(np.argmax(scores.mean(axis=1)))
            output.append(dict(id=row["id"], tier=row["tier"], target=row["digit"], guess=guess,
                               correct=int(guess == row["digit"])))
    with (ROOT / "results/probe.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print("External five-examples-per-digit control:", sum(r["correct"] for r in output), "/", len(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("tokens")
    p = sub.add_parser("probe")
    p.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    if args.mode == "tokens":
        count_tokens()
    else:
        activation_probe(args.data)
