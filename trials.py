"""Stage lossless text packets and record responses from isolated Codex subagents."""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".packets"
MODELS = ["gpt-6.1-terra", "gpt-6-luna"]
TIERS = ["linear", "h4", "h32", "h32x32", "h128x64"]


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def jsonline(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def packets(data, limit=14000):
    block = bytearray()
    for line in data.splitlines(keepends=True):
        if len(line) > limit:
            raise ValueError("oversized row")
        if block and len(block) + len(line) > limit:
            yield bytes(block)
            block.clear()
        block.extend(line)
    if block:
        yield bytes(block)


def stage_one(uid, data):
    assert data.isascii() and len(data) <= 300_000
    directory = STAGE / uid
    directory.mkdir(exist_ok=True, parents=True)
    blocks = list(packets(data))
    assert b"".join(blocks) == data
    receipts = []
    for n, block in enumerate(blocks, 1):
        receipt = hashlib.sha256(uid.encode() + str(n).encode() + block).hexdigest()[:8]
        receipts.append(receipt)
        (directory / f"{n:02}.txt").write_bytes(block + f"\n[packet {n}/{len(blocks)} receipt {receipt}]\n".encode())
    return dict(id=uid, parts=len(blocks), receipts=receipts, sha256=hashlib.sha256(data).hexdigest())


def stage():
    groups = {t: [] for t in ["edit", *TIERS]}
    for folder in ["fixtures", "edits"]:
        for row in lines(ROOT / folder / "manifest.jsonl"):
            uid = row["id"]
            tier = row["tier"] if folder == "fixtures" else "edit"
            data = (ROOT / folder / "prompts" / (uid + ".txt")).read_bytes()
            metadata = stage_one(uid, data)
            groups[tier].append(dict(id=uid, task="edit" if tier == "edit" else "digit", tier=tier,
                                     parts=metadata["parts"], receipts=metadata["receipts"], sha256=metadata["sha256"]))
    for group in groups.values():
        group.sort(key=lambda r: r["id"])
        random.Random(19112026).shuffle(group)
    assert [len(g) for g in groups.values()] == [4, 10, 10, 10, 10, 10]
    schedule = [g[i] for i in range(10) for g in groups.values() if i < len(g)]
    (ROOT / "schedule.jsonl").write_text("".join(map(jsonline, schedule)))


def task(row):
    path = STAGE / row["id"]
    return f"""Independent blind trial. Read the assigned {row['parts']} packet(s) in numerical order, paths {path}/01.txt through {path}/{row['parts']:02}.txt. For transport only, use functions.exec with exactly one cat per packet, e.g. // @exec: {{"max_output_tokens":10000}} on line 1, then text((await tools.exec_command({{cmd:"cat {path}/01.txt", max_output_tokens:16000}})).output). Use no other commands, files, analysis tools, code or agents. Read every packet fully. Infer the answer yourself from the displayed text. Keep this task in one context. If a packet is truncated, report delivery failure; if context was compacted, report delivery failure. Return one JSON object with "answer" containing the exact object requested in the assigned prompt and "receipts" containing the final eight-character receipt from each packet in order. No explanation."""


def has_receipts(raw, expected):
    """Read the delivery field independently of the syntax of the submitted answer."""
    matches = re.findall(r'"receipts"\s*:\s*(\[[^\[\]]*\])', raw)
    if len(matches) != 1:
        return False
    try:
        return json.loads(matches[0]) == expected
    except ValueError:
        return False


def record(model, uid, agent, raw, status, schedule=None, results_dir=None):
    row = next(x for x in lines(schedule or ROOT / "schedule.jsonl") if x["id"] == uid)
    dest = (results_dir or ROOT / "results") / model / (uid + ".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    delivered = status == "completed" and has_receipts(raw, row["receipts"])
    payload = dict(model=model, effort="medium", id=uid, agent=agent, status=status,
                   delivery_verified=delivered, prompt_sha256=row["sha256"], raw=raw)
    with dest.open("x") as file:
        file.write(jsonline(payload))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="cmd", required=True)
    commands.add_parser("stage")
    t = commands.add_parser("task")
    t.add_argument("id")
    r = commands.add_parser("record")
    r.add_argument("model", choices=MODELS)
    r.add_argument("id")
    r.add_argument("agent")
    r.add_argument("--status", choices=["completed", "transport_failure", "compacted", "tool_violation"], default="completed")
    a = p.parse_args()
    if a.cmd == "stage":
        stage()
    elif a.cmd == "task":
        print(task(next(x for x in lines(ROOT / "schedule.jsonl") if x["id"] == a.id)))
    else:
        record(a.model, a.id, a.agent, sys.stdin.read(), a.status)
