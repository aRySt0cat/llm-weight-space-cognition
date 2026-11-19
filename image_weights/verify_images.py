"""Read back every tensor from the rendered PNG and check its rounding and predictions."""

import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from study import HERE, ROOT, digest
from prepare import dataset_path, logits, parse_prompt, pool_images
from trials import lines


def runs(mask):
    indices = np.flatnonzero(mask)
    return [
        (int(g[0]), int(g[-1]))
        for g in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        if g.size
    ]


def decode(rgb, n_in, n_out, scale):
    height, width = rgb.shape[:2]
    assert height % n_out == 0 and width % n_in == 0
    dy, dx = height // n_out, width // n_in
    tile = rgb[::dy, ::dx]
    assert np.array_equal(np.repeat(np.repeat(tile, dy, axis=0), dx, axis=1), rgb)
    red, green, blue = tile.astype(int).transpose(2, 0, 1)
    assert np.all(np.maximum(red, blue) == 255) and np.all(
        green == np.minimum(red, blue)
    )
    return (red - blue) * (scale / 255)


def verify(data):
    with np.load(dataset_path(data)) as archive:
        x, y = pool_images(archive["x_test"]), archive["y_test"]
    fixtures = {r["id"]: r for r in lines(ROOT / "fixtures/manifest.jsonl")}
    results = []
    with threadpool_limits(limits=1):
        for row in lines(HERE / "schedule.jsonl"):
            path = HERE / "images" / (row["id"] + ".png")
            assert digest(path.read_bytes()) == row["sha256"]
            with Image.open(path) as im:
                pixels = np.asarray(im)
                assert not im.info
            fixture = fixtures[row["source"]]
            layers = parse_prompt(
                (ROOT / "fixtures/prompts" / (row["source"] + ".txt")).read_text()
            )
            border = np.all(pixels == 128, axis=2)
            bands = runs(border[:, 15])
            restored = []
            for (w, b), (top, bottom) in zip(layers, bands, strict=True):
                (left, right), (bl, br) = runs(border[top])
                n_in, n_out = w.shape
                printed_scale = float(
                    format(max(np.abs(w).max(), np.abs(b).max()) or 1.0, ".5g")
                )
                inside = pixels[top + 1 : bottom]
                rw = decode(inside[:, left + 1 : right], n_in, n_out, printed_scale).T
                rb = decode(inside[:, bl + 1 : br], 1, n_out, printed_scale).ravel()
                for original, decoded in ((w, rw), (b, rb)):
                    assert (
                        np.max(np.abs(original - decoded))
                        <= printed_scale / 510 + 1e-12
                    )
                restored.append((rw, rb))
            original, decoded = (logits(net, x) > 0 for net in (layers, restored))
            target = y == fixture["digit"]
            results.append(
                dict(
                    id=row["id"],
                    source=row["source"],
                    tier=row["tier"],
                    params=fixture["params"],
                    decoded_scalars=sum(w.size + b.size for w, b in restored),
                    same_decision=int((original == decoded).sum()),
                    test_images=len(y),
                    text_balanced_accuracy=balanced_accuracy_score(target, original),
                    image_balanced_accuracy=balanced_accuracy_score(target, decoded),
                )
            )
    with (HERE / "encoding_check.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    print(
        "Predictions preserved",
        sum(r["same_decision"] for r in results),
        "/",
        sum(r["test_images"] for r in results),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)

    verify(Path(parser.parse_args().data))
