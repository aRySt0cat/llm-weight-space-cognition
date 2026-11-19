"""Re-score the stored, unedited responses and verify all input/output associations."""
import csv
from collections import Counter
from functools import cache
import hashlib
import json
from math import factorial

from editing import read_answer, score
from trials import ROOT, MODELS, TIERS, has_receipts, lines

OUT = ROOT / "results"


def permutation_p(guesses, hits):
    """Exact one-sided tail for a random bijection to ten distinct target digits."""
    assert len(guesses) == 10

    @cache
    def count(used, needed):
        index = used.bit_count()
        if needed <= 0:
            return factorial(10 - index)
        if needed > 10 - index:
            return 0
        return sum(count(used | (1 << digit), needed - int(guesses[index] == digit))
                   for digit in range(10) if not used & (1 << digit))

    return count(0, hits) / factorial(10)


def holm(ps):
    result = {}
    previous = 0
    for i, (tier, p) in enumerate(sorted(ps.items(), key=lambda x: x[1])):
        previous = max(previous, min(1, (len(ps) - i) * p))
        result[tier] = previous
    return result


def global_permutation_p(rows):
    """Relabel the ten digit identities once for all tiers, preserving repeated-digit dependence."""
    table = [[0] * 10 for _ in range(10)]
    for row in rows:
        if row["guess"] is not None:
            table[row["target"]][row["guess"]] += 1
    hits = sum(table[d][d] for d in range(10))

    @cache
    def tail(used, needed):
        i = used.bit_count()
        if needed <= 0:
            return factorial(10 - i)
        if i == 10:
            return 0
        return sum(tail(used | (1 << j), needed - table[i][j]) for j in range(10) if not used & (1 << j))

    return tail(0, hits) / factorial(10)


def write_csv(name, records):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)


def analyze():
    schedule = lines(ROOT / "schedule.jsonl")
    digits = {x["id"]: x for x in lines(ROOT / "fixtures/manifest.jsonl")}
    edits = {x["id"]: x for x in lines(ROOT / "edits/manifest.jsonl")}
    sizes = {x["id"]: x for x in lines(OUT / "sizes.jsonl")}
    evaluated = []
    used_agents = set()
    for trial in schedule:
        uid = trial["id"]
        folder = "fixtures" if uid in digits else "edits"
        source = (ROOT / folder / "prompts" / (uid + ".txt")).read_bytes()
        assert hashlib.sha256(source).hexdigest() == trial["sha256"]
        assert sizes[uid]["bytes"] == len(source) <= 300_000
        for model in MODELS:
            record = dict(id=uid, model=model, tier=trial["tier"], params=digits[uid]["params"] if uid in digits else 30,
                          o200k_base=sizes[uid]["o200k_base"], target=digits[uid]["digit"] if uid in digits else None,
                          status="not_run", receipts_match=False, delivered=False, valid_answer=False, guess=None, correct=False,
                          excluded_guess=None, excluded_would_match=None,
                          correct_values=None, max_abs_error=None)
            evaluated.append(record)
            path = OUT / model / (uid + ".json")
            if not path.exists():
                continue
            raw = json.loads(path.read_text())
            assert (raw["model"], raw["id"], raw["prompt_sha256"]) == (model, uid, trial["sha256"])
            assert raw["agent"] not in used_agents, f"Reused respondent {raw['agent']}"
            used_agents.add(raw["agent"])
            receipt_match = has_receipts(raw["raw"], trial["receipts"])
            delivery = raw["status"] == "completed" and receipt_match
            assert raw["delivery_verified"] == delivery
            record.update(status=raw["status"], receipts_match=receipt_match, delivered=delivery)
            try:
                envelope = read_answer(raw["raw"])
                answer = envelope.get("answer") if isinstance(envelope, dict) else None
            except ValueError:
                answer = None
            digit_answer = (isinstance(answer, dict) and set(answer) == {"digit"}
                           and type(answer["digit"]) is int and 0 <= answer["digit"] <= 9)
            if not delivery:
                if uid in digits and digit_answer:
                    record.update(excluded_guess=answer["digit"], excluded_would_match=answer["digit"] == record["target"])
                continue
            if uid in edits:
                result = score(edits[uid], answer)
                record.update(valid_answer=result["error"] is None, correct=result["passed"],
                              correct_values=result["correct_values"], max_abs_error=result["max_abs_error"])
            elif digit_answer:
                record.update(valid_answer=True, guess=answer["digit"], correct=answer["digit"] == record["target"])
    write_csv("trials.csv", evaluated)
    stats = []
    for model in MODELS:
        by_tier = {}
        for tier in TIERS:
            rows = [r for r in evaluated if r["model"] == model and r["tier"] == tier]
            received = [r for r in rows if r["delivered"]]
            hits = sum(r["correct"] for r in received)
            assert len(rows) == 10 and len({r["target"] for r in rows}) == 10
            p = permutation_p([r["guess"] for r in received], hits) if len(received) == 10 else None
            entry = dict(model=model, tier=tier, params=rows[0]["params"], assigned=len(rows),
                         recorded=sum(r["status"] != "not_run" for r in rows), delivered=len(received),
                         malformed=sum(not r["valid_answer"] for r in received), correct=hits, perm_p=p, holm_p=None)
            stats.append(entry)
            by_tier[tier] = entry
        adjusted = holm({t: 1.0 if v["perm_p"] is None else v["perm_p"] for t, v in by_tier.items()})
        for tier, entry in by_tier.items():
            if entry["perm_p"] is not None:
                entry["holm_p"] = adjusted[tier]
    write_csv("recognition.csv", stats)
    pooled = []
    for model in MODELS:
        rows = [r for r in evaluated if r["model"] == model and r["tier"] in TIERS and r["delivered"]]
        valid = sum(r["valid_answer"] for r in rows)
        pooled.append(dict(model=model, eligible=len(rows), valid_digits=valid,
                           correct=sum(r["correct"] for r in rows), expected_under_permutation=valid / 10,
                           exploratory_p=global_permutation_p(rows)))
    adjusted = holm({r["model"]: r["exploratory_p"] for r in pooled})
    for row in pooled:
        row["holm_two_models"] = adjusted[row["model"]]
    write_csv("pooled_exploratory.csv", pooled)
    # Secondary sensitivity analysis. Include observed outbound-only tool deviations
    # with complete receipts; never promote these cases to the primary result.
    outbound_ids = {(r["model"], r["id"]) for r in lines(OUT / "deviations.jsonl") if r["type"] == "outbound_only"}
    sensitivity = []
    for model in MODELS:
        for tier in TIERS:
            rows = [r for r in evaluated if r["model"] == model and r["tier"] == tier]
            accepted = [r for r in rows if r["delivered"]]
            outbound = [r for r in rows if r["status"] == "tool_violation" and r["receipts_match"]
                        and (model, r["id"]) in outbound_ids]
            guesses = [r["guess"] for r in accepted] + [r["excluded_guess"] for r in outbound]
            hits = sum(r["correct"] for r in accepted) + sum(r["excluded_would_match"] is True for r in outbound)
            sensitivity.append(dict(model=model, tier=tier, primary_n=len(accepted), extra_n=len(outbound),
                                    total_n=len(guesses), correct=hits,
                                    perm_p=permutation_p(guesses, hits) if len(guesses) == 10 else None))
    write_csv("sensitivity.csv", sensitivity)
    paired = []
    for tier in [*TIERS, "edit"]:
        groups = {m: {r["id"]: r for r in evaluated if r["model"] == m and r["tier"] == tier and r["delivered"]}
                  for m in MODELS}
        left, right = groups.values()
        common = left.keys() & right.keys()
        counts = Counter((left[k]["correct"], right[k]["correct"]) for k in common)
        paired.append(dict(tier=tier, common=len(common), both_correct=counts[True, True],
                           terra_only=counts[True, False], luna_only=counts[False, True], both_failed=counts[False, False]))
    write_csv("paired.csv", paired)
    editing = []
    for model in MODELS:
        rows = [r for r in evaluated if r["model"] == model and r["tier"] == "edit" and r["delivered"]]
        editing.append(dict(model=model, assigned=4, delivered=len(rows), correct=sum(r["correct"] for r in rows),
                            correct_values=sum(r["correct_values"] for r in rows), total_values=32*len(rows)))
    write_csv("editing.csv", editing)
    print("recorded", len(used_agents), "/", len(evaluated))
    for m in MODELS:
        print(m, [(r["tier"], r["correct"], r["delivered"]) for r in stats if r["model"] == m])
    print(editing)
    return stats, editing


if __name__ == "__main__":
    analyze()
