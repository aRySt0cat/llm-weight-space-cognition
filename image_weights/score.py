"""Score the first image responses and compare with the original text responses."""

import csv
from functools import cache
import json
from math import factorial
import re

import numpy as np
from scipy.stats import binomtest
from study import HERE, ROOT, MODELS, digest
from analyze import holm
from editing import read_answer
from trials import TIERS, lines


def write_table(name, rows):
    with (HERE / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def parse(raw, receipt):
    matches = re.findall(r'"receipt"\s*:\s*"([^"]*)"', raw)
    try:
        answer = read_answer(raw)
    except ValueError:
        answer = None
    guess = None
    if (
        isinstance(answer, dict)
        and set(answer) == {"digit", "receipt"}
        and type(answer["digit"]) is int
        and 0 <= answer["digit"] <= 9
    ):
        guess = answer["digit"]
    return matches == [receipt], guess


def distribution(targets, guesses):
    # Count label permutations with k matches; targets can be a subset after exclusions.
    @cache
    def counts(mask):
        position = mask.bit_count()
        if position == len(targets):
            return (1,)
        out = [0] * (len(targets) - position + 1)
        for j, digit in enumerate(targets):
            if mask & (1 << j):
                continue
            hit = int(guesses[position] == digit)
            for k, count in enumerate(counts(mask | (1 << j))):
                out[k + hit] += count
        return tuple(out)

    result = counts(0)
    assert sum(result) == factorial(len(targets))
    return result


def chance_test(rows):
    dist = np.array([1], dtype=object)
    for t in TIERS:
        tier = [r for r in rows if r["tier"] == t and r["eligible"]]
        counts = distribution(
            tuple(r["target"] for r in tier), tuple(r["guess"] for r in tier)
        )
        dist = np.convolve(dist, np.array(counts, dtype=object))
    hits = sum(r["image_correct"] for r in rows)
    return float(sum(dist[hits:]) / sum(dist))


def evaluate():
    checkpoints = {r["id"]: r for r in lines(ROOT / "fixtures/manifest.jsonl")}
    with (ROOT / "results/trials.csv").open() as f:
        text_results = {
            (r["model"], r["id"]): r for r in csv.DictReader(f) if r["tier"] in TIERS
        }
    deviations = {
        (r["model"], r["id"]): r["note"]
        for path in HERE.glob("deviations-*.jsonl")
        for r in lines(path)
    }
    schedule = lines(HERE / "schedule.jsonl")
    assert {r["source"] for r in schedule} == set(checkpoints) and len(schedule) == 50
    rows = []
    old_agents = {
        json.loads((ROOT / "results" / model / (uid + ".json")).read_text())["agent"]
        for model, uid in text_results
    }
    agents = set()
    for case in schedule:
        uid = case["id"]
        saved_image = (HERE / "images" / (uid + ".png")).read_bytes()
        assert digest(saved_image) == case["sha256"]
        source = case["source"]
        checkpoint = checkpoints[source]
        assert case["tier"] == checkpoint["tier"]
        for model in MODELS:
            response = json.loads(
                (HERE / "responses" / model / (uid + ".json")).read_text()
            )
            agent = response["agent"]
            assert agent not in agents and agent not in old_agents
            agents.add(agent)
            assert (
                response["model"],
                response["id"],
                response["image_sha256"],
                response["effort"],
            ) == (model, uid, case["sha256"], "medium")
            receipt_ok, guess = parse(response["raw"], case["receipt"])
            deviation = deviations.get((model, uid), "")
            if deviation:
                status = "tool_violation"
            elif response["raw"] == "DELIVERY_FAILURE":
                status = "reported_delivery_failure"
            elif not receipt_ok:
                status = "unverified_receipt"
            else:
                status = "received"
            eligible = status == "received"
            target = checkpoint["digit"]
            text = text_results[model, source]
            assert int(text["target"]) == target
            rows.append(
                dict(
                    id=uid,
                    source=source,
                    tier=case["tier"],
                    params=checkpoint["params"],
                    model=model,
                    target=target,
                    status=status,
                    receipt_ok=receipt_ok,
                    deviation=deviation,
                    eligible=eligible,
                    guess=guess,
                    image_correct=bool(eligible and guess == target),
                    text_eligible=text["delivered"] == "True",
                    text_guess=text["guess"],
                    text_correct=text["correct"] == "True",
                )
            )
    write_table("trials.csv", rows)
    sizes, paired = [], []
    for model in MODELS:
        group = [r for r in rows if r["model"] == model]
        for tier in TIERS:
            cluster = [r for r in group if r["tier"] == tier]
            assert len(cluster) == 10
            sizes.append(
                dict(
                    model=model,
                    tier=tier,
                    params=cluster[0]["params"],
                    assigned=len(cluster),
                    image_eligible=sum(r["eligible"] for r in cluster),
                    image_correct=sum(r["image_correct"] for r in cluster),
                    text_eligible=sum(r["text_eligible"] for r in cluster),
                    text_correct=sum(r["text_correct"] for r in cluster),
                )
            )
        common = [r for r in group if r["eligible"] and r["text_eligible"]]
        text_only = sum(r["text_correct"] and not r["image_correct"] for r in common)
        image_only = sum(r["image_correct"] and not r["text_correct"] for r in common)
        p = (
            binomtest(image_only, image_only + text_only).pvalue
            if image_only + text_only
            else 1.0
        )
        paired.append(
            dict(
                model=model,
                assigned=len(group),
                image_eligible=sum(r["eligible"] for r in group),
                image_valid_digits=sum(
                    r["eligible"] and r["guess"] is not None for r in group
                ),
                image_correct=sum(r["image_correct"] for r in group),
                common=len(common),
                both=sum(r["image_correct"] and r["text_correct"] for r in common),
                text_only=text_only,
                image_only=image_only,
                neither=sum(
                    not r["image_correct"] and not r["text_correct"] for r in common
                ),
                paired_p=p,
                chance_p=chance_test(group),
            )
        )
    for key in ("paired_p", "chance_p"):
        adj = holm({r["model"]: r[key] for r in paired})
        for r in paired:
            r[key + "_holm"] = adj[r["model"]]
    write_table("sizes.csv", sizes)
    write_table("paired.csv", paired)
    for r in paired:
        print(r)


if __name__ == "__main__":
    evaluate()
