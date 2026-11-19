"""Render the frozen MNIST checkpoints as layer matrices and stage blind trials."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from prepare import parse_prompt
from trials import lines

FONT = ImageFont.load_default(size=15)
MODELS = ("gpt-6.2-astra", "gpt-6.1-terra", "gpt-6-luna")
PROMPT = (HERE / "prompt.txt").read_text().strip()
NEAREST = Image.Resampling.NEAREST


def digest(data):
    return hashlib.sha256(data).hexdigest()


def colors(values, scale):
    # 255 signed levels on each side of white; red/blue channels retain the bin.
    levels = np.rint(255 * np.abs(values) / scale).astype(np.uint8)
    light = 255 - levels
    red = np.where(values < 0, light, 255)
    blue = np.where(values < 0, 255, light)
    return np.stack([red, light, blue], axis=-1).astype(np.uint8)


def render(layers, receipt):
    layout = []
    for weights, bias in layers:
        n_in, n_out = weights.shape
        cell_width = min(12, 784 // n_in)
        cell_height = max(4, min(18, cell_width)) if n_out > 1 else 18
        layout.append((weights, bias, cell_width, cell_height))
    height = 24 + sum(w.shape[1] * dy + 54 for w, _, _, dy in layout) + 24
    canvas = Image.new("RGB", (865, height), "#f5f5f5")
    draw = ImageDraw.Draw(canvas)
    y = 14
    for index, (weights, bias, dx, dy) in enumerate(layout, 1):
        n_in, n_out = weights.shape
        scale = max(np.abs(weights).max(), np.abs(bias).max()) or 1.0
        draw.text(
            (16, y),
            f"Layer {index}    W: {n_in} inputs x {n_out} outputs    scale: {scale:.5g}",
            font=FONT,
            fill="#222222",
        )
        w_right = 16 + dx * n_in
        if w_right > 480:
            draw.text((w_right + 14, y), "b", font=FONT, fill="#222222")
        y += 23
        for x, data, width in [
            (16, weights.T, dx * n_in),
            (w_right + 14, bias[:, None], 14),
        ]:
            tile = Image.fromarray(colors(data, scale)).resize(
                (width, dy * n_out), NEAREST
            )
            canvas.paste(tile, (x, y))
            draw.rectangle(
                (x - 1, y - 1, x + width, y + dy * n_out), outline="#808080", width=1
            )
        y += dy * n_out + 31
    draw.text((16, height - 26), f"Receipt: {receipt}", font=FONT, fill="#111111")
    return canvas


def stage():
    out = HERE / "images"
    out.mkdir(exist_ok=True)
    schedule = []
    for case in lines(ROOT / "fixtures/manifest.jsonl"):
        data = (ROOT / "fixtures/prompts" / (case["id"] + ".txt")).read_bytes()
        assert digest(data) == case["sha256"]
        uid = digest(b"matrix-image-v1:" + data)[:16]
        receipt = digest(b"matrix-receipt:" + uid.encode())[:8]
        path = out / (uid + ".png")
        render(parse_prompt(data.decode()), receipt).save(path, optimize=False)
        schedule.append(
            dict(
                id=uid,
                source=case["id"],
                tier=case["tier"],
                receipt=receipt,
                sha256=digest(path.read_bytes()),
            )
        )
    schedule.sort(key=lambda r: r["id"])
    assert (
        len({r["id"] for r in schedule}) == len({r["receipt"] for r in schedule}) == 50
    )
    (HERE / "schedule.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in schedule)
    )


def task(uid):
    path = HERE / "images" / (uid + ".png")
    return (
        f"Independent blind trial. For transport, use functions.exec exactly once: "
        f'image((await tools.view_image({{path:"{path}"}})).image_url). '
        "Inspect the returned image visually. Use no other tools, files, commands, agents or messages. "
        "If the image is unavailable return DELIVERY_FAILURE.\n\n" + PROMPT
    )


def record(model, uid, agent, raw):
    schedule = {r["id"]: r for r in lines(HERE / "schedule.jsonl")}
    dest = HERE / "responses" / model / (uid + ".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    row = dict(
        model=model,
        effort="medium",
        id=uid,
        agent=agent,
        image_sha256=schedule[uid]["sha256"],
        raw=raw.strip(),
    )
    with dest.open("x") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage")
    q = sub.add_parser("task")
    q.add_argument("id")
    q = sub.add_parser("record")
    q.add_argument("model", choices=MODELS)
    q.add_argument("id")
    q.add_argument("agent")
    a = parser.parse_args()
    if a.cmd == "stage":
        stage()
    elif a.cmd == "task":
        print(task(a.id))
    elif a.cmd == "record":
        record(a.model, a.id, a.agent, sys.stdin.read())
