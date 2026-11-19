"""Build one standalone Japanese slide deck from the scored records and official MNIST images."""

import argparse
import base64
import csv
from collections import Counter
from html import escape
from io import BytesIO
import json
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


def weight_map(weights, scale, digit, row, col):
    relative = weights / scale
    neutral = np.array([248, 247, 240])
    color = np.where(relative[:, :, None] >= 0, [166, 60, 32], [37, 83, 153])
    rgb = np.rint(neutral + (color - neutral) * np.abs(relative[:, :, None])).astype(
        np.uint8
    )
    return (
        f'<svg class="weight-map" viewBox="0 0 140 140" role="img" aria-label="「{digit}」分類器の重み。上から{row + 1}行、左から{col + 1}列目を囲む">'
        f'<image href="{png(rgb)}" width="140" height="140" style="image-rendering:pixelated"/>'
        f'<rect x="{col * 10}" y="{row * 10}" width="10" height="10" fill="none" stroke="#15272e" stroke-width="1.7"/></svg>'
    )


def net_svg():
    xs = [50, 203, 350, 490]
    layers = [[30, 57, 105, 132], [27, 54, 108, 135], [30, 57, 105, 132], [81]]
    out = [
        '<svg class="net" viewBox="0 0 540 178" role="img" aria-label="全結合ネットワーク。入力196、隠れ層128と64、出力1。各層は一部のユニットを表示">'
    ]
    for i in range(3):
        for a in layers[i]:
            for b in layers[i + 1]:
                out.append(
                    f'<path d="M{xs[i]} {a}L{xs[i + 1]} {b}" stroke="#a0b4ad" stroke-width="1.1"/>'
                )
    for i, ys in enumerate(layers):
        for y in ys:
            color = "#006858" if i == 3 else "#ffffff"
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


def matrix_table(headers, rows):
    thead = "".join(f'<th scope="col">{h}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>"


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


def paired_accuracy_chart(paired):
    parts = [
        '<svg class="paired-bars" viewBox="0 0 550 245" role="img"><title>共通問題での正答率と正答数</title>'
    ]
    for i, (model, name) in enumerate(zip(MODELS, SHORT)):
        row = paired[model]
        common = int(row["common"])
        y = 21 + 79 * i
        parts.append(
            f'<text x="0" y="{y + 19}" fill="#172b32" font-size="18">{name}</text>'
        )
        for j, (key, color) in enumerate(
            (("text_only", "#64758c"), ("image_only", "#006858"))
        ):
            correct = int(row["both"]) + int(row[key])
            width = 320 * correct / common if common else 0
            bar_y = y + 22 * j - 10
            parts.extend(
                [
                    f'<rect x="110" y="{bar_y}" width="320" height="16" fill="#e5e9e7"/>',
                    f'<rect x="110" y="{bar_y}" width="{width:.2f}" height="16" fill="{color}"/>',
                    f'<text x="440" y="{bar_y + 14}" font-size="18" fill="#172b32">{ratio(correct, common)}</text>',
                ]
            )
    return "".join(parts) + "</svg>"


def add_image_slides(add, params, draft):
    # The image condition uses the same checkpoint as the running text example.
    folder = ROOT / "image_weights"

    def image_table(name):
        with (folder / name).open() as f:
            return list(csv.DictReader(f))

    case_answers = ""
    if not draft or (folder / "trials.csv").is_file():
        sample = {
            r["model"]: r for r in image_table("trials.csv") if r["source"] == EXAMPLE
        }
        guesses = "　".join(
            f"{name}「{sample[m]['guess'] if sample[m]['eligible'] == 'True' and sample[m]['guess'] else '無効・除外'}」"
            for m, name in zip(MODELS, SHORT)
        )
        case_answers = (
            f'<p class="packet-answers"><b>この「7」の分類器への回答</b>　{guesses}</p>'
        )
    packet = next(r for r in lines(folder / "schedule.jsonl") if r["source"] == EXAMPLE)
    image_path = folder / "images" / (packet["id"] + ".png")
    matrix_png = "data:image/png;base64," + base64.b64encode(
        image_path.read_bytes()
    ).decode("ascii")
    # Original layer 1 starts at x=16, y=37, with 4x4 display pixels per coefficient.
    with Image.open(image_path) as im:
        zoom = png(np.asarray(im.crop((16, 37, 16 + 42 * 4, 37 + 12 * 4))))
    add(
        "画像入力　実験設定",
        f"""<h2>今度は、全層の重みを画像で渡す</h2>
<p class="intro">同じ50個の分類器を使用。各モデル・各問で新しいコンテキストを用意し、画像と読み方だけを提示した。</p>
<div class="visual-protocol"><figure class="whole-packet"><a href="{REPO}/blob/main/image_weights/images/{packet["id"]}.png"><img src="{matrix_png}" alt="実際に渡した3層分の重みとバイアス"></a><figcaption>33,537パラメータの出題画像（クリックで拡大）</figcaption></figure>
<div><h3>第1層は、入力196列 × 出力128行</h3><p>1行が1ユニットの重み。右端の別列がバイアス。入力は14 × 14画素を行順に並べた。</p>
<figure class="matrix-zoom"><img src="{zoom}" alt="第1層左上、42入力・12出力の重みの拡大"><figcaption>第1層左上の42列 × 12行</figcaption></figure>
<div class="color-legend"><i></i><span>負</span><span>0</span><span>正</span></div>
<p>各層で重みとバイアスに共通の色尺度を使い、負・正を各255段階で表した。</p></div></div>{case_answers}""",
    )

    if draft and not all(
        (folder / name).is_file() for name in ("sizes.csv", "paired.csv")
    ):
        return

    sized = {(r["model"], r["tier"]): r for r in image_table("sizes.csv")}
    paired = {r["model"]: r for r in image_table("paired.csv")}
    size_rows = []
    for tier in TIERS:
        cells = [f"{params[tier]:,}"]
        for model in MODELS:
            r = sized[model, tier]
            cells.append(
                f'<span class="baseline">{r["text_correct"]}/{r["text_eligible"]}</span> → <b>{r["image_correct"]}/{r["image_eligible"]}</b>'
            )
        size_rows.append(cells)
    image_total = "　　".join(
        f"{name} <b>{paired[m]['image_correct']}/{paired[m]['image_eligible']}</b>"
        for m, name in zip(MODELS, SHORT)
    )
    linear = sized[MODELS[0], "linear"]
    hidden_hits = int(paired[MODELS[0]]["image_correct"]) - int(linear["image_correct"])
    hidden_n = int(paired[MODELS[0]]["image_eligible"]) - int(linear["image_eligible"])
    add(
        "画像入力　構成別の結果",
        f"""<h2>画像での Astra は線形 {linear["image_correct"]}/{linear["image_eligible"]}、隠れ層あり {hidden_hits}/{hidden_n}</h2>
<p class="intro">各セルは <span class="baseline">数値列</span> → <b>画像</b>。分子は正答数、分母はその条件で手順に適合した問題数。</p>
<div class="matrix-scores">{matrix_table(["パラメータ数", *SHORT], size_rows)}</div>
<p class="finding">画像での合計　{image_total}</p>
<p class="fine">Luna の画像条件で除外した3問は、画像の受領失敗2問と追加送信1問。全割当を分母にすると4/50。 <a href="{REPO}/blob/main/image_weights/trials.csv">1問ごとの回答</a></p>""",
    )
    paired_rows = []
    for model, name in zip(MODELS, SHORT):
        r = paired[model]
        paired_rows.append(
            [
                name,
                r["text_only"],
                r["image_only"],
                f"{float(r['paired_p_holm']):.3g}",
            ]
        )
    add(
        "画像入力　同じ問題どうしの比較",
        f"""<h2>同じ問題で、数値列と画像の正答を比べる</h2>
<p class="intro">両条件で採点できた分類器に絞ると、Astra は数値列38/50、画像10/50。Terra と Luna では有意な差は出なかった。</p>
<div class="pairing"><div><h3>共通問題での正答数 / 問題数</h3><div class="paired-key"><span>数値列</span><span>画像</span></div>{paired_accuracy_chart(paired)}</div>
<div><h3>片方の条件だけで正答した問題数</h3>{matrix_table(["モデル", "数値列のみ", "画像のみ", "補正 p 値"], paired_rows)}
<p class="fine">両側の正確 McNemar 検定。3モデルで Holm 補正。各問・各条件につき1回答。</p></div></div>
<p class="bottom-note">数値列は14画素ごとに改行し、画像は196画素を横1列に配置した。色への丸めも加わるため、比較の対象はこの2つの表示方法。 <a href="{REPO}/blob/main/image_weights/REPORT.md">集計結果</a></p>""",
    )


def build(archive, output, draft):
    manifest = lines(ROOT / "fixtures/manifest.jsonl")
    recog = {(r["model"], r["tier"]): r for r in csv_rows("recognition")}
    trials = [r for r in csv_rows("trials") if r["tier"] in TIERS]
    schedule = [r for r in lines(ROOT / "schedule.jsonl") if r["task"] == "digit"]
    if not draft:
        assert {(r["model"], r["id"]) for r in trials if r["status"] != "not_run"} == {
            (m, case["id"]) for case in schedule for m in MODELS
        }, "Score all MNIST first passes before exporting"
    by_case = {(r["model"], r["id"]): r for r in trials}
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
        slides.append((section, content, kind))

    add(
        "2026.11.19　勉強会",
        f"""<div class="cover-layout"><div>
<h1>LLM は<em>重み</em>から<br>数字を読めるか</h1>
<p class="intro">「7」とそれ以外を見分けるよう学習したネット。<br>重みを数値列や画像で示すと、「7」と答えられるか。</p>
<p class="cover-models">GPT-6.2-astra　/　GPT-6.1-terra　/　GPT-6-luna</p></div>
<figure class="digit-plate"><div class="digit-grid">{"".join(examples)}</div><figcaption>MNIST のテスト画像　28 × 28 画素</figcaption></figure></div>""",
        "cover",
    )

    add(
        "実験設定",
        f"""<h2>画像で学習し、重みの数値列を出題する</h2>
<p class="intro">各分類器が学習するのは「ある数字」と「それ以外」を分ける二値分類。</p>
<div class="flow"><figure class="center"><div class="image-tile">{examples[7]}</div><figcaption>28 × 28 画素</figcaption></figure><div class="flow-arrow">→</div>
<figure class="center"><div class="image-tile">{ink_image(pooled, "同じ7を2×2の平均で14×14に縮小")}</div><figcaption>14 × 14 画素<br>（2 × 2 の平均）</figcaption></figure><div class="flow-arrow">→</div>
<figure>{net_svg()}</figure></div>
<div class="data-strip"><div class="sample"><h3>LLM に与える情報</h3><pre>widths 196 128 64 1
{escape(excerpt)}</pre><p class="caption">層の幅と数値列の抜粋。実験では全重み・全バイアスを提示。</p></div>
<div class="question"><h3>どの数字を検出する？</h3><p>回答は 0〜9 の1個。<br>LLMには画像や出力例を渡さない。</p></div></div>
<p class="bottom-note">各設問には、インクが 1、背景が 0 であることと、画素・重みの並び順も記した。</p>""",
    )

    arch_rows = [
        (
            WIDTHS[i],
            f"{params[t]:,}",
            f"{100 * mean(r['balanced_accuracy'] for r in tier_data[t]):.1f}%",
        )
        for i, t in enumerate(TIERS)
    ]
    add(
        "実験設定",
        f"""<h2>3モデルに、同じ50個の分類器を出題</h2>
<p class="intro">数字0〜9について、幅と層数の異なる5構成を学習。各モデルは1つの分類器につき1回答。</p>
<div class="setup"><div><h3>学習したネットワーク</h3>{matrix_table(["各層の幅", "パラメータ数", "分類器の性能¹"], arch_rows)}
<p class="fine">¹ 公式テスト画像1万枚での感度と特異度の平均。各構成10個の平均値。</p></div>
<div class="conditions"><h3>各問の条件</h3><dl>
<dt>学習画像</dt><dd>対象数字5,000枚と、それ以外5,000枚。</dd>
<dt>LLMへの入力</dt><dd>英語の設問と全パラメータ。毎回新しいコンテキストで、正解や過去の回答は渡さない。</dd>
<dt>数値列条件の回答方法</dt><dd>推論設定は medium。許可した操作は問題文の表示のみ。共有環境での全操作の独立監査は未実施。</dd></dl></div></div>
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
        f"""<h2>重みの数値列から対象数字を当てた割合</h2>
<p class="intro">棒は正答率。棒の上は「正答数 / 手順に適合した問題数」。各構成の割当は10問。</p>
<div class="chart-top"><div class="legend">{legend}</div><p>破線は10択の偶然正答率</p></div>
{accuracy_chart(recog, params)}<div class="totals"><strong>全サイズ合計</strong>{overall}</div>
<p class="fine">横軸はパラメータ数。{significance}（片側置換検定、Holm補正5%）。<br>形式が無効な回答は誤答に数えた。6-luna の追加送信による除外6件も採点すると 5/50。</p>""",
    )

    example_case = next(r for r in schedule if r["id"] == EXAMPLE)
    case_rows = []
    for model, name in zip(MODELS, SHORT, strict=True):
        record = by_case[model, EXAMPLE]
        answer = json.dumps({"digit": int(record["guess"])}) if record["guess"] else "—"
        link = f"{REPO}/blob/main/results/{model}/{EXAMPLE}.json"
        case_rows.append([name, f'<a href="{link}"><code>{escape(answer)}</code></a>'])
    add(
        "考察　保存された回答",
        f"""<h2>同じ「7」の分類器に、3通りの回答</h2>
<p class="intro">p.02 の33,537パラメータの例。各モデルが受け取った数値列と設問は同じ。</p>
<div class="case"><figure><div class="image-tile">{examples[7]}</div><figcaption>学習対象の数字（解説用）</figcaption></figure>
<div>{matrix_table(["モデル", "最終回答の digit 欄"], case_rows)}
<p class="fine">3モデルとも、全{example_case["parts"]}片の受領符号が一致。表の回答から元の記録を開ける。</p></div></div>
<p class="finding">出題時に「数字を1つ、理由は書かずに」と指定した。数値列での150件の最終回答には、候補を比較した経過や判断理由の説明はない。</p>
<p class="sources"><a href="{REPO}/blob/main/fixtures/prompts/{EXAMPLE}.txt">この分類器の設問と全パラメータ</a>　<a href="{REPO}/blob/main/trials.py">回答形式の指示</a></p>""",
    )

    # Compare the same central pixel across two linear checkpoints on one color scale.
    row, col = 6, 7
    linear = [r for r in manifest if r["tier"] == "linear" and r["digit"] in (0, 8)]
    linear.sort(key=lambda r: r["digit"])
    maps = [
        parse_prompt((ROOT / "fixtures/prompts" / (r["id"] + ".txt")).read_text())[0][
            0
        ].reshape(14, 14)
        for r in linear
    ]
    scale = max(np.abs(w).max() for w in maps)
    figures = []
    for r, w in zip(linear, maps, strict=True):
        digit = r["digit"]
        value = f"{w[row, col]:+.2f}".replace("-", "−")
        figures.append(
            f'<figure><div class="map-label">「{digit}」分類器</div>'
            f"{weight_map(w, scale, digit, row, col)}<figcaption>枠の重み <b>{value}</b></figcaption></figure>"
        )
    comparison = []
    for model, name in zip(MODELS, SHORT, strict=True):
        cells = []
        for r in linear:
            trial = by_case[model, r["id"]]
            cells.append(
                trial["guess"] or ("無効" if trial["delivered"] == "True" else "—")
            )
        comparison.append([name, *cells])
    add(
        "考察　数値列にある手掛かり",
        f"""<h2>「0」と「8」では、中央の画素が逆に効く</h2>
<p class="intro">197パラメータの線形分類器では、196個の重みが14 × 14の画素と1対1に対応する。</p>
<div class="maps-and-answers"><div><div class="map-pair">{"".join(figures)}</div>
<p class="map-key"><i class="neg"></i>負 <i class="pos"></i>正　色の濃さは重みの大きさ（共通尺度）</p></div>
<div><h3>この2問での回答</h3>{matrix_table(["モデル", "対象 0", "対象 8"], comparison)}
<p class="fine">枠は上から7行目、左から8列目。他の画素を固定してここにインクを足すと、「0」のスコアは下がり、「8」のスコアは上がる。</p></div></div>
<p class="bottom-note">数値列の実験では、この図は LLM に見せていない。この位置を LLM が判断に使ったかは、回答記録からは分からない。</p>""",
    )

    astra = {
        (r["tier"], int(r["target"])): r for r in trials if r["model"] == MODELS[0]
    }
    grid = []
    for tier in TIERS:
        cells = [f"{params[tier]:,}"]
        for digit in range(10):
            trial = astra[tier, digit]
            css = "hit" if trial["correct"] == "True" else "miss"
            cells.append(
                f'<span class="prediction {css}">{trial["guess"] or "—"}</span>'
            )
        grid.append(cells)
    add(
        "考察　対象数字と回答",
        f"""<h2>Astra は「8」を5構成のうち1構成で当てた</h2>
<p class="intro">列が学習対象、マスの数字が実際の回答。それぞれ独立した設問。</p>
<div class="matrix-key"><span><i class="hit"></i>正答</span><span><i class="miss"></i>誤答</span></div>
<div class="digit-matrix">{matrix_table(["パラメータ数 ／ 対象", *range(10)], grid)}</div>
<p class="finding">「8」への回答は小さい構成から 2、8、2、4、7。「0」と「6」は、全5構成で正答した。</p>
<p class="fine">最大の33,537パラメータでも5問正解した。各数字・各構成につき1個のネットを試したため、読めるサイズの上限は定まらない。</p>""",
    )

    frequencies = []
    for model, name in zip(MODELS, SHORT, strict=True):
        valid = [
            r["guess"]
            for r in trials
            if r["model"] == model
            and r["valid_answer"] == "True"
            and r["delivered"] == "True"
        ]
        counts = Counter(valid)
        frequencies.append([name, *(counts[str(d)] for d in range(10)), len(valid)])
    add(
        "考察　回答の偏りと記録の限界",
        f"""<h2>Terra は50問中14問で「1」と回答</h2>
<p class="intro">各数字が正解になる設問は5問ずつ。表は、各モデルが有効な回答でその数字を選んだ回数。</p>
<div class="freq">{matrix_table(["回答した数字", *range(10), "合計"], frequencies)}</div>
<div class="observations"><div><h3>Terra の「1」</h3><p>対象が「1」の5問は全問正答。ほかの数字を学習した分類器にも、9回「1」と答えた。</p></div>
<div><h3>Luna の回答形式</h3><p>適合44問中、39問で有効な数字を返した。残る5問は <code>?</code> が4件、JSONを2つに分けた回答が1件。</p></div></div>
<p class="fine">別の6件では Luna が運営役に追加送信し、返信はなかった。この6件は集計から除外した。　<a href="{REPO}/blob/main/results/deviations.jsonl">送信の記録</a></p>
<p class="bottom-note">回答の偏りは観測できる。その数字を選ぶまでに、ほかの候補を検討したかは記録されていない。　<a href="{REPO}/blob/main/results/trials.csv">採点表</a></p>""",
    )

    add_image_slides(add, params, draft)

    template = (HERE / "template.html").read_text()
    markup = "\n".join(
        f'<section class="slide {kind}" aria-label="{n}枚目 {section}">{content}'
        f'<footer class="foot"><span>{section}</span><span class="pagenum">{n:02d} / {len(slides):02d}</span></footer></section>'
        for n, (section, content, kind) in enumerate(slides, start=1)
    )
    output.write_text(template.replace("<!--SLIDES-->", markup), encoding="utf-8")


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
