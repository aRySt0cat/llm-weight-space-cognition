"""Build one standalone Japanese slide deck from the scored records and official MNIST images."""

import argparse
import base64
import csv
from html import escape
from io import BytesIO
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from prepare import dataset_path, parse_prompt
from trials import TIERS, lines

MODELS = ["gpt-6.2-astra", "gpt-6.1-terra", "gpt-6-luna"]
SHORT = ["6.2-astra", "6.1-terra", "6-luna"]
COLORS = ["#006858", "#4e69ac", "#ac572e"]
WIDTHS = [
    "196 → 1",
    "196 → 4 → 1",
    "196 → 32 → 1",
    "196 → 32 → 32 → 1",
    "196 → 128 → 64 → 1",
]
EXAMPLE = "0bd23458ac578ebf"  # The target-7 case used in the original two-model report.
REPO = "https://github.com/aRySt0cat/llm-weight-space-cognition"


def csv_rows(name):
    with (ROOT / "results" / (name + ".csv")).open() as f:
        return list(csv.DictReader(f))


def ratio(hits, n):
    return f"{hits}/{n}" if n else "—"


def png(array):
    output = BytesIO()
    Image.fromarray(array).save(output, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(
        "ascii"
    )


def ink_image(pixels, alt):
    # Preserve the original intensities as opacity, with the same bright-ink polarity.
    rgba = np.full((*pixels.shape, 4), 255, dtype=np.uint8)
    rgba[:, :, 3] = pixels
    return f'<img class="ink" src="{png(rgba)}" alt="{escape(alt)}" width="{pixels.shape[1]}" height="{pixels.shape[0]}">'


def weight_image(manifest):
    uid = next(r["id"] for r in manifest if r["tier"] == "linear" and r["digit"] == 7)
    w = parse_prompt((ROOT / "fixtures/prompts" / (uid + ".txt")).read_text())[0][
        0
    ].reshape(14, 14)
    w = w / np.abs(w).max()
    neutral = np.array([248, 247, 240])
    color = np.where(w[:, :, None] >= 0, [162, 64, 36], [40, 86, 154])
    arr = np.rint(neutral + (color - neutral) * np.abs(w[:, :, None])).astype(np.uint8)
    return f'<img src="{png(arr)}" width="14" height="14" alt="7用の線形分類器。赤は正の重み、青は負の重み">'


def net_svg(focus=False):
    xs = [50, 203, 350, 490]
    layers = [[30, 57, 105, 132], [27, 54, 108, 135], [30, 57, 105, 132], [81]]
    out = [
        '<svg class="net" viewBox="0 0 540 178" role="img" aria-label="全結合ネットワーク。入力196、隠れ層128と64、出力1。各層は一部のユニットを表示">'
    ]
    for i in range(3):
        stroke = "#006858" if focus and i == 2 else "#a0b4ad"
        for a in layers[i]:
            for b in layers[i + 1]:
                out.append(
                    f'<path d="M{xs[i]} {a}L{xs[i + 1]} {b}" stroke="{stroke}" stroke-width="{2 if focus and i == 2 else 1.1}"/>'
                )
    for i, ys in enumerate(layers):
        for y in ys:
            color = "#006858" if i == 3 or (i == 2 and focus) else "#ffffff"
            out.append(
                f'<circle cx="{xs[i]}" cy="{y}" r="6.6" fill="{color}" stroke="#407366" stroke-width="1.7"/>'
            )
        if len(ys) > 1:
            for y in [75, 81, 87]:
                out.append(f'<circle cx="{xs[i]}" cy="{y}" r="1.4" fill="#6a817b"/>')
        lab = ["入力 196", "ReLU 128", "ReLU 64", "出力 1"][i]
        out.append(
            f'<text x="{xs[i]}" y="166" fill="#28403f" font-size="17" text-anchor="middle">{lab}</text>'
        )
    return "".join(out) + "</svg>"


def segment_svg():
    polygons = [
        "24,8 90,8 97,15 87,22 28,22 16,15",
        "99,18 105,24 97,79 90,85 83,78 91,28",
        "90,89 97,95 90,148 80,155 75,148 82,98",
        "73,151 80,158 72,165 10,165 4,158 14,151",
        "8,93 16,87 22,94 15,147 5,154 0,147",
        "18,19 26,27 19,76 13,84 7,77 13,26",
        "23,80 79,80 86,87 76,94 22,94 15,87",
    ]
    return (
        '<svg viewBox="0 0 108 174" role="img" aria-label="上と右の2本が点灯した7セグメントの数字7">'
        + "".join(
            f'<polygon points="{p}" fill="{("#d1efcc" if i < 3 else "#375152")}"/>'
            for i, p in enumerate(polygons)
        )
        + "</svg>"
    )


def matrix_table(headers, rows):
    thead = "".join(f'<th scope="col">{h}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>"


def model_name(index):
    return f'<span class="model-name">{SHORT[index]}</span>'


def accuracy_chart(recog, params):
    max_pct = max(
        (
            100 * int(r["correct"]) / int(r["delivered"])
            for r in recog.values()
            if int(r["delivered"])
        ),
        default=0,
    )
    ceiling = 50 if max_pct <= 43 else 100
    top, bottom = 28, 262
    y = lambda pct: bottom - (bottom - top) * pct / ceiling
    desc = "。".join(
        f"{params[t]} パラメータ "
        + "、".join(
            f"{s} {recog[m, t]['correct']}/{recog[m, t]['delivered']}"
            for m, s in zip(MODELS, SHORT)
        )
        for t in TIERS
    )
    out = [
        f'<svg class="chart" viewBox="0 0 1120 307" role="img"><title>モデル別の数字同定正答率</title><desc>{desc}</desc>'
    ]
    ticks = list(range(0, ceiling + 1, 10 if ceiling == 50 else 20))
    for tick in ticks:
        if tick == 10:
            continue
        v = y(tick)
        out.append(
            f'<path d="M60 {v}H1110" stroke="#ccd5d0"/><text x="50" y="{v + 5}" text-anchor="end" font-size="16" fill="#52676c">{tick}%</text>'
        )
    v = y(10)
    out.append(
        f'<path d="M60 {v}H1110" stroke="#5b6866" stroke-width="1.7" stroke-dasharray="5 5"/><text x="50" y="{v + 5}" text-anchor="end" font-size="16" fill="#52676c">10%</text>'
    )
    for i, tier in enumerate(TIERS):
        mid = 164 + 209 * i
        out.append(
            f'<text x="{mid}" y="294" text-anchor="middle" fill="#172b32" font-size="21">{params[tier]:,}</text>'
        )
        for j, model in enumerate(MODELS):
            row = recog[model, tier]
            n, correct = int(row["delivered"]), int(row["correct"])
            pct = 100 * correct / n if n else 0
            x = mid + 51 * (j - 1)
            h = bottom - y(pct)
            if n:
                out.append(
                    f'<rect x="{x - 14}" y="{bottom - max(2, h)}" width="28" height="{max(2, h)}" fill="{COLORS[j]}"/>'
                )
            out.append(
                f'<text x="{x}" y="{y(pct) - 8}" text-anchor="middle" font-size="19" fill="{COLORS[j]}" font-weight="650">{ratio(correct, n)}</text>'
            )
    return "".join(out) + "</svg>"


def build(archive, output, draft):
    manifest = lines(ROOT / "fixtures/manifest.jsonl")
    recog = {(r["model"], r["tier"]): r for r in csv_rows("recognition")}
    edits = {r["model"]: r for r in csv_rows("editing")}
    trials, inversions, segment = (
        csv_rows(n) for n in ["trials", "inversion", "segment"]
    )
    primary_schedule = lines(ROOT / "schedule.jsonl")
    inv_schedule = lines(ROOT / "inversion/schedule.jsonl")
    seg_schedule = lines(ROOT / "segment/schedule.jsonl")
    if not draft:
        for collection, schedule in (
            (trials, primary_schedule),
            (inversions, inv_schedule),
            (segment, seg_schedule),
        ):
            assert {
                (r["model"], r["id"]) for r in collection if r["status"] != "not_run"
            } == {(m, case["id"]) for case in schedule for m in MODELS}, (
                "Score all first passes before exporting"
            )
    tier_data = {t: [r for r in manifest if r["tier"] == t] for t in TIERS}
    params = {t: rr[0]["params"] for t, rr in tier_data.items()}
    with np.load(dataset_path(archive), allow_pickle=False) as data:
        labels, images = data["y_test"], data["x_test"]
        indices = [int(np.flatnonzero(labels == d)[0]) for d in range(10)]
        pixels = [images[i] for i in indices]
    examples = [
        ink_image(x, f"MNISTテスト画像、数字{d}、テスト配列のインデックス{indices[d]}")
        for d, x in enumerate(pixels)
    ]
    pooled = pixels[7].reshape(14, 2, 14, 2).mean(axis=(1, 3)).round().astype(np.uint8)
    case_path = ROOT / "fixtures/prompts" / (EXAMPLE + ".txt")
    weight_lines = case_path.read_text().split("layer 1\n", 1)[1].splitlines()
    excerpt = "\n".join(
        " ".join(line.split()[2:8]) + "  …" for line in weight_lines[3:5]
    )
    slides = []

    def add(section, content, kind=""):
        n = len(slides) + 1
        slides.append(
            f'<section class="slide {kind}" aria-label="{n}枚目 {section}">{content}<footer class="foot"><span>{section}</span><span class="pagenum">{n:02d} / 08</span></footer></section>'
        )

    add(
        "2026.11.19　勉強会",
        f"""<div class="cover-layout"><div>
<h1>LLM は<em>重み</em>から<br>数字を読めるか</h1>
<p class="intro">画像への応答からは、分類器の機能を調べられる。<br>重みだけで、その機能を推定・編集できるか。</p>
<p class="cover-models">GPT-6.2-astra　/　GPT-6.1-terra　/　GPT-6-luna</p></div>
<figure class="digit-plate"><div class="digit-grid">{"".join(examples)}</div><figcaption>MNIST のテスト画像　28 × 28 画素</figcaption></figure></div>""",
        "cover",
    )

    add(
        "実験設定",
        f"""<h2>画像で学習し、重みだけを出題する</h2>
<p class="intro">各分類器が学習するのは「ある数字」と「それ以外」を分ける二値分類。</p>
<div class="flow"><figure class="center"><div class="image-tile">{examples[7]}</div><figcaption>28 × 28 画素</figcaption></figure><div class="flow-arrow">→</div>
<figure class="center"><div class="image-tile">{ink_image(pooled, "同じ7を2×2の平均で14×14に縮小")}</div><figcaption>14 × 14 画素<br>（2 × 2 の平均）</figcaption></figure><div class="flow-arrow">→</div>
<figure>{net_svg()}</figure></div>
<div class="data-strip"><div class="sample"><h3>LLM に与える情報</h3><pre>widths 196 128 64 1
{escape(excerpt)}</pre><p class="caption">層の幅と数値列の抜粋。実験では全重み・全バイアスを提示。</p></div>
<div class="question"><h3>どの数字を検出する？</h3><p>回答は 0〜9 の1個。<br>回答役に画像や推論例は渡さない。</p></div></div>
<p class="bottom-note">画像を各数字5枚ずつ入力し、平均ロジットが最大の数字を選ぶと、分類器50個すべての対象を特定できた。</p>""",
    )

    arch_rows = [
        (
            WIDTHS[i],
            f"{params[t]:,}",
            f"{100 * mean(r['balanced_accuracy'] for r in tier_data[t]):.1f}%",
        )
        for i, t in enumerate(TIERS)
    ]
    tasks = [
        ("MNIST の対象数字を同定", 50),
        ("7セグメントの対象数字を同定", 10),
        ("4ビット入力の局所編集", 4),
        ("MNIST の出力層を編集", 8),
    ]
    add(
        "実験設定",
        f"""<h2>3モデルに、同じ72問を出題</h2>
<p class="intro">各問で新しいコンテキストを開始。推論設定は medium。許可したツール操作は問題文の表示のみ。</p>
<div class="setup"><div><h3>学習済み MNIST 分類器</h3>{matrix_table(["各層の幅", "パラメータ数", "分類器の性能¹"], arch_rows)}
<p class="fine">¹ 公式テスト画像1万枚での balanced accuracy。各構成10個の平均。</p></div>
<div><h3>1モデルあたりの内訳</h3><div class="task-list">{"".join(f'<div class="task-item"><span>{a}</span><b>{b}問</b></div>' for a, b in tasks)}</div>
<p class="methods">MNIST の同定は5構成 × 数字10種。<br>各数字につき正例5,000枚・負例5,000枚で学習。<br>共有ファイル環境で、全操作の独立監査は未実施。</p></div></div>
<p class="bottom-note">重みは有効数字最大5桁。最大の問題文は292,204バイト（参考値 130,669 トークン）。</p>""",
    )

    total = {
        m: (
            sum(int(recog[m, t]["correct"]) for t in TIERS),
            sum(int(recog[m, t]["delivered"]) for t in TIERS),
        )
        for m in MODELS
    }
    legend = "".join(
        f'<span><i class="key" style="background:{c}"></i>{s}</span>'
        for c, s in zip(COLORS, SHORT)
    )
    astra_tests = [
        float(recog[MODELS[0], t]["holm_p"])
        for t in TIERS
        if recog[MODELS[0], t]["holm_p"]
    ]
    significance = (
        f"Astra は検定できた{len(astra_tests)}構成中{sum(p < 0.05 for p in astra_tests)}構成で偶然水準を上回った"
        if astra_tests
        else "Astra のサイズ別検定は集計後に実施"
    )
    if len(astra_tests) == len(TIERS) and all(p < 0.05 for p in astra_tests):
        significance = f"Astra は全{len(TIERS)}構成で偶然水準を上回った"
    overall = "".join(
        f'<span class="total">{s}<b>{ratio(*total[m])}</b></span>'
        for m, s in zip(MODELS, SHORT)
    )
    add(
        "数字の同定　MNIST",
        f"""<h2>重みから対象数字を当てた割合</h2>
<p class="intro">棒は正答率。棒の上は「正答数 / 手順に適合した問題数」。各構成の割当は10問。</p>
<div class="chart-top"><div class="legend">{legend}</div><p>破線は10択の偶然正答率</p></div>
{accuracy_chart(recog, params)}<div class="totals"><strong>全サイズ合計</strong>{overall}</div>
<p class="fine">横軸はパラメータ数。{significance}（片側置換検定、Holm補正5%）。<br>無効回答は誤答。6-luna の許可外の送信6件まで含めると 5/50。</p>""",
    )

    seg_rows = []
    for j, m in enumerate(MODELS):
        good = [r for r in segment if r["model"] == m and r["eligible"] == "True"]
        linear = recog[m, "linear"]
        seg_rows.append(
            [
                model_name(j),
                ratio(sum(r["correct"] == "True" for r in good), len(good)),
                ratio(int(linear["correct"]), int(linear["delivered"])),
            ]
        )
    add(
        "数字の同定　対照課題",
        f"""<h2>入力の意味が分かる7セグメント表示</h2>
<p class="intro">各入力が対応する位置を明示し、7個の重みとバイアスから対象数字を答えさせた。</p>
<div class="control"><div><div class="visual-pair">
<figure><div class="seg-tile">{segment_svg()}</div><figcaption>7セグメントの「7」</figcaption></figure>
<figure><div class="map-tile">{weight_image(manifest)}</div><figcaption>MNIST の「7」の重み<br>赤は正、青は負</figcaption></figure></div>
<p class="weight-vector">+2　+2　+2　−2　−2　−2　−2</p>
<p class="caption">上・右上・右下が +2。バイアスは −5。</p></div>
<div>{matrix_table(["正答 / 適合", "7セグメント<br>8パラメータ", "MNIST 線形<br>197パラメータ"], seg_rows)}</div></div>
<p class="bottom-note">2つの課題では、入力表現も重みの構造も異なる。</p>""",
    )

    edit_rows = [
        [
            model_name(j),
            ratio(int(edits[m]["correct"]), int(edits[m]["delivered"])),
            ratio(int(edits[m]["correct_values"]), int(edits[m]["total_values"])),
        ]
        for j, m in enumerate(MODELS)
    ]
    add(
        "数値編集　4ビット入力",
        f"""<h2>一方の出力だけで、1ビットを反転する</h2>
<p class="intro">30パラメータのネットを4個用意。出力層の数値を最大2個まで変更できる。</p>
<div class="edit-grid"><div class="patch"><h3>1問目の編集例</h3><p class="formula">−2x₃ → −2 + 2x₃</p>
<p class="caption">隠れユニットの出力は 1.25x₃</p>
<div class="patch-row"><span>出力の重み</span><b>−1.6 → +1.6</b></div>
<div class="patch-row"><span>バイアス</span><b>7 → 5</b></div></div>
<div>{matrix_table(["モデル", "成功 / 適合", "合致した出力値"], edit_rows)}
<p class="fine">1問につき全16入力 × 2出力。全32値が許容誤差0.001以内で成功。</p></div></div>
<p class="bottom-note">6-luna の失敗2問は、重みの符号を反転し、必要なバイアス変更を省略していた。</p>""",
    )

    inv_rows = []
    for j, m in enumerate(MODELS):
        row = [model_name(j)]
        for t in ("h4", "h128x64"):
            data = [r for r in inversions if r["model"] == m and r["tier"] == t]
            row.append(
                ratio(
                    sum(
                        r["delivered"] == "True" and r["exact"] == "True" for r in data
                    ),
                    len(data),
                )
            )
        inv_rows.append(row)
    add(
        "数値編集　MNIST",
        f"""<h2>最終層を編集し、出力の符号を反転する</h2>
<p class="intro">すべての入力でロジットを −1 倍にする。最終層の全重みとバイアスを数値で提出。</p>
<div class="readout"><figure>{net_svg(True)}<div class="formula">w · h + b → (−w) · h − b</div></figure>
<div>{matrix_table(["モデル", "793パラメータ<br>5値を編集", "33,537パラメータ<br>65値を編集"], inv_rows)}
<p class="fine">手順に適合し全値一致した数 / 初回の割当数。<br>提出値を代入して、テスト画像1万枚でも照合。</p></div></div>
<p class="bottom-note">最大構成の6-luna は、ほかに2件が全値一致・許可外送信、1件が未完。未完の1件は独立再試行で全値一致。</p>""",
    )

    inv_id = next(r["id"] for r in inv_schedule if r["source_id"] == EXAMPLE)
    case_rows = []
    for j, m in enumerate(MODELS):
        rr = next(r for r in trials if r["id"] == EXAMPLE and r["model"] == m)
        ir = next(r for r in inversions if r["id"] == inv_id and r["model"] == m)
        guess = (
            rr["guess"]
            if rr["guess"]
            else ("集計中" if rr["status"] == "not_run" else "除外・無効")
        )
        patch = (
            f"{ir['matching_scalars']}/65"
            if ir["valid_answer"] == "True"
            else ("集計中" if ir["status"] == "not_run" else "除外・無効")
        )
        case_rows.append(
            [model_name(j), guess, f'<span class="case-result">{patch}</span>']
        )
    add(
        "考察",
        f'''<h2>同じ「7」の分類器に、2通りの問い</h2>
<p class="intro">33,537パラメータの同一チェックポイント。数字の同定と最終層の編集は別のコンテキストで実施。</p>
<div class="case"><figure><div class="image-tile">{examples[7]}</div><figcaption>学習した対象は「7」</figcaption></figure>
{matrix_table(["モデル", "推定した数字", "編集で一致した数値"], case_rows)}</div>
<p class="finding">最終層の符号反転は、その層の式から解ける。編集の成功だけでは、対象数字を読み取れたかは分からない。</p>
<p class="fine">Astra の数字同定は最大構成で {recog[MODELS[0], "h128x64"]["correct"]}/{recog[MODELS[0], "h128x64"]["delivered"]}。各サイズ10問・1数字1ネットのため、認識できる規模の上限は定まらない。</p>
<p class="sources"><a href="{REPO}">実験コード・データ</a>　 <a href="{REPO}/blob/main/results/REPORT.md">採点結果・除外理由</a></p>''',
    )

    assert len(slides) == 8
    template = (HERE / "template.html").read_text()
    output.write_text(
        template.replace("<!--SLIDES-->", "\n".join(slides)), encoding="utf-8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data", required=True, type=Path, help="Path to the official Keras MNIST npz"
    )
    p.add_argument("--output", type=Path, default=HERE / "index.html")
    p.add_argument(
        "--draft",
        action="store_true",
        help="Render a layout preview during data collection",
    )
    args = p.parse_args()
    build(args.data, args.output, args.draft)
