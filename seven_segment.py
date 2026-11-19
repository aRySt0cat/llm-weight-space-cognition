"""Generate and score the supplementary seven-segment recognition calibration."""
import argparse
import csv
import hashlib
import itertools
import random
import sys

import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from editing import read_answer
from trials import MODELS, ROOT, has_receipts, jsonline, lines, record, stage_one, task


DIRECTORY = ROOT / "segment"
SCHEDULE = DIRECTORY / "schedule.jsonl"
MANIFEST = DIRECTORY / "manifest.jsonl"
SEGMENTS = "abcdefg"
PATTERNS = ("abcdef", "bc", "abdeg", "abcdg", "bcfg",
            "acdfg", "acdefg", "abc", "abcdefg", "abcdfg")
STATES = np.array(list(itertools.product((0, 1), repeat=7)), dtype=np.float64)
ID_SEED = 237480912
ASSIGNMENT_SEED = 971639284
SCHEDULE_SEED = 486023715
PROMPT = """This trained binary classifier recognizes one decimal digit on a seven-segment display. Which digit (0 through 9) does it recognize? Return only a JSON object of the form {"digit": n}. Use no tools, code execution, calculators, or external aids.

Each input is 1 if its segment is lit and 0 if it is dark. In order, the segments are a (top), b (upper right), c (lower right), d (bottom), e (lower left), f (upper left), and g (middle). The score is the bias plus the dot product of the weights and inputs in this order. A score greater than zero means the classifier recognizes the pattern.

"""


def assignments():
    id_rng = random.Random(ID_SEED)
    ids = set()
    while len(ids) < 10:
        ids.add(f"{id_rng.getrandbits(64):016x}")
    digits = list(range(10))
    random.Random(ASSIGNMENT_SEED).shuffle(digits)
    return list(zip(sorted(ids), digits, strict=True))


def labels(digit):
    target = [int(s in PATTERNS[digit]) for s in SEGMENTS]
    return (STATES == target).all(axis=1)


def make_prompt(digit):
    model = SVC(kernel="linear", C=1000, tol=1e-8).fit(STATES, labels(digit))
    weights = " ".join(format(float(w), ".5g") for w in model.coef_[0])
    bias = format(float(model.intercept_[0]), ".5g")
    return (PROMPT + f"weights {weights}\nbias {bias}\n").encode("ascii")


def verify_prompt(data, digit):
    prefix, weights, bias, end = data.decode("ascii").rsplit("\n", 3)
    if end or prefix + "\n" != PROMPT or not weights.startswith("weights ") or not bias.startswith("bias "):
        raise ValueError("Invalid seven-segment prompt")
    coefficients = np.array([float(w) for w in weights.split()[1:]])
    if coefficients.shape != (7,) or len(bias.split()) != 2:
        raise ValueError("Expected seven weights and one bias")
    scores = STATES @ coefficients + float(bias.split()[1])
    target = labels(digit)
    if not np.isfinite(scores).all() or int(target.sum()) != 1 or not np.array_equal(scores > 0, target):
        raise ValueError(f"Rounded checkpoint for digit {digit} failed exhaustive verification")
    return int(len(scores))


def metadata(uid, digit, data):
    return dict(id=uid, digit=digit, lit_segments=PATTERNS[digit], params=8,
                states_correct=verify_prompt(data, digit), states_total=len(STATES),
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def stage():
    prompts = DIRECTORY / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    rows, manifest = {}, []
    with threadpool_limits(limits=1):
        for uid, digit in assignments():
            data = make_prompt(digit)
            manifest.append(metadata(uid, digit, data))
            (prompts / f"{uid}.txt").write_bytes(data)
            rows[uid] = stage_one(uid, data)
    SCHEDULE.write_text("".join(jsonline(rows[uid]) for uid in schedule_ids()))
    MANIFEST.write_text("".join(map(jsonline, manifest)))
    print(f"Staged {len(rows)} classifiers. All {len(rows) * len(STATES)} rounded-checkpoint decisions verified.")


def score_response(response, row, model, target):
    result = dict(id=row["id"], model=model, params=8, target=target, status="not_run",
                  receipts_match=None, eligible=False, answer_status="unscored", guess=None, correct=None)
    if response is None:
        return result
    if (response["id"], response["model"], response["prompt_sha256"]) != (row["id"], model, row["sha256"]):
        raise ValueError("Response identity does not match its trial")
    matched = has_receipts(response["raw"], row["receipts"])
    eligible = response["status"] == "completed" and matched
    if eligible != response["delivery_verified"]:
        raise ValueError("Response delivery flag disagrees with protocol")
    result.update(status=response["status"], receipts_match=matched, eligible=eligible)
    if not eligible:
        return result
    try:
        envelope = read_answer(response["raw"])
    except ValueError:
        result.update(answer_status="invalid_json", correct=False)
        return result
    answer = envelope.get("answer") if isinstance(envelope, dict) else None
    if (not isinstance(envelope, dict) or set(envelope) != {"answer", "receipts"}
            or envelope["receipts"] != row["receipts"]
            or not isinstance(answer, dict) or set(answer) != {"digit"}
            or type(answer["digit"]) is not int or not 0 <= answer["digit"] <= 9):
        result.update(answer_status="invalid_schema", correct=False)
        return result
    result.update(answer_status="valid", guess=answer["digit"], correct=answer["digit"] == target)
    return result


def score():
    manifest = {r["id"]: r for r in lines(MANIFEST)}
    schedule = lines(SCHEDULE)
    expected = dict(assignments())
    if [r["id"] for r in schedule] != schedule_ids() or set(expected) != set(manifest):
        raise ValueError("Seven-segment control allocation changed")
    results = []
    used_agents = set()
    for row in schedule:
        uid = row["id"]
        data = (DIRECTORY / "prompts" / f"{uid}.txt").read_bytes()
        target = expected[uid]
        current = metadata(uid, target, data)
        if current != manifest[uid] or row["sha256"] != current["sha256"]:
            raise ValueError(f"Fixture integrity check failed for {uid}")
        for model in MODELS:
            file = ROOT / "results" / model / f"{uid}.json"
            response = read_answer(file.read_text()) if file.exists() else None
            if response is not None:
                if response["agent"] in used_agents:
                    raise ValueError(f"Reused respondent {response['agent']}")
                used_agents.add(response["agent"])
            results.append(score_response(response, row, model, target))
    destination = ROOT / "results/segment.csv"
    destination.parent.mkdir(exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    print(f"Scored {sum(r['eligible'] for r in results)} eligible trials across {len(results)} assignments.")


def schedule_ids():
    ids = [uid for uid, _ in assignments()]
    random.Random(SCHEDULE_SEED).shuffle(ids)
    return ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("stage")
    commands.add_parser("score")
    commands.add_parser("task").add_argument("id")
    recorder = commands.add_parser("record")
    recorder.add_argument("model", choices=MODELS)
    recorder.add_argument("id")
    recorder.add_argument("agent")
    recorder.add_argument("--status", default="completed",
                          choices=("completed", "transport_failure", "compacted", "tool_violation"))
    args = parser.parse_args()
    if args.command == "stage":
        stage()
    elif args.command == "score":
        score()
    elif args.command == "task":
        print(task(next(row for row in lines(SCHEDULE) if row["id"] == args.id)))
    else:
        record(args.model, args.id, args.agent, sys.stdin.read(), args.status, SCHEDULE)
