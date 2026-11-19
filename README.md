# LLM は重みから MNIST 分類器の対象数字を読めるか

MNIST 分類器の対象数字を、重みの数値列から答えられたのは **gpt-6.2-astra が38/50、gpt-6.1-terra が10/50、gpt-6-luna が5/44**。同じ72問を各モデルの独立コンテキストで実施した。数字の同定50問、7セグメント同定10問、4ビット入力の編集4問、MNIST の出力層編集8問。推論設定はすべて medium。

[勉強会スライド（HTML）](slides/index.html) ・ [3モデルの結果と考察](results/REPORT.md) ・ [全問の採点表](results/trials.csv) ・ [事前手順](PROTOCOL.md) ・ [Astra の追加条件](ASTRA_PROTOCOL.md)

スライドは画像とSVGを埋め込んだ単一のHTMLファイル。ブラウザで開き、左右キーで移動、Fキーで全画面表示。印刷時は1枚につき1ページになる。[HTMLをダウンロード](https://github.com/aRySt0cat/llm-weight-space-cognition/raw/refs/heads/main/slides/index.html)

## 重みを画像で渡す追加実験

同じ50個の MNIST 分類器について、重み行列とバイアスを層ごとの赤・白・青の画像にし、新しいコンテキストで3モデルに対象数字を答えさせた。画像での正答は Astra **10/50**、Terra **8/50**、Luna **4/47**。最大構成の第1層は入力196列 × 出力128行。[画像条件と数値列条件の比較](image_weights/REPORT.md) ・ [50枚の出題画像](image_weights/images/) ・ [事前に固定した条件](image_weights/PROTOCOL.md)

## 数字の同定

MNIST の画像を 2×2 の平均プーリングで 14×14 にし、0〜9 の各数字について「その数字か、それ以外か」を学習した。すべての構成で同一の入力表現を使い、同じ数字の二値分類器には同じ訓練例を与えた。LLM の入力は匿名の設問、画素と出力の定義、全バイアスと全重み。入力画像や推論例は与えない。各重みを有効数字最大 5 桁、先頭の小数ゼロ省略ありの ASCII テキストに変換した。最初の層では 14 画素ごとに改行した。

| ネットワーク | パラメータ | 問題文のバイト数 | 参考トークン数 |
|---|---:|---:|---:|
| 196→1 | 197 | 2,282–2,393 | 868–940 |
| 196→4→1 | 793 | 7,063–7,472 | 2,977–3,323 |
| 196→32→1 | 6,337 | 52,564–54,554 | 22,632–25,159 |
| 196→32→32→1 | 7,393 | 61,673–63,370 | 26,493–27,901 |
| 196→128→64→1 | 33,537 | 285,692–292,204 | 122,941–130,669 |

バイト数には設問とテンソルの区切りも含む。ASCII の各記号をすべて 1 トークンと数えても 30 万以内に収まる。参考値は `o200k_base` で計数した。実行モデル固有のトークン使用量は取得していない。チャットのシステム指示、配信時の受領符号、思考・出力はこの表の対象外。

## 8パラメータの数字同定（追加対照）

7セグメント表示で、7個の重みとバイアスから、どの数字を検出する線形判定器かを答える。同じ10択でも入力の各次元に「右上」「中央」などの直接的な意味がある。各数字の判定器を全128点灯パターンで学習・照合した。全10問を追加実験の回答収集前にコミット `4e41f92` で固定した。[生成方法・全問](segment/README.md)

## 数値編集

4ビットの入力、4個の ReLU 隠れユニット、2出力のネットワークを4個用意した。それぞれ30パラメータで、入力ビットと隠れユニットの対応、向き、尺度が異なる。出力層は最小二乗で学習した。課題は出力の一方だけで指定ビットを反転した場合と同じ値を返すよう、出力層のスカラー最大2個を置き換えること。変更後のネットを全16入力で実行し、両出力の全32値が許容誤差0.001以内に収まれば成功とした。[生成方法・対照・採点仕様](edits/README.md)

## MNIST の出力層編集（追加実験）

793・33,537パラメータの既存分類器で、全入力に対して出力ロジットが元の符号反転となるよう、出力層の具体的な重みとバイアスを提出させた。両サイズとも対象数字0・2・5・7を使う。変更対象はそれぞれ5値・65値。全値の完全一致を判定し、提出値を代入したネットワークをテスト画像1万枚でも検証した。問題はこの追加実験の回答収集前にコミット `351d77f` で固定した。[出題・採点条件](inversion/README.md)

## 回答側と採点側

問題と正解を固定したコミットは `e00c37d`。各試行に親チャットの履歴を継承しない新規サブエージェントを割り当て、試行間の会話、正誤のフィードバック、他の試行の回答は与えない。回答側に許可したツール操作は、割り当てられたファイルのテキストを `cat` で表示することのみ。数値の計算、重みの検索、図による表示、正解表の閲覧は指示で禁じた。採点側は別途 Python を使用した。

配信時は長い問題を元の行順のまま最大14,000バイトごとに分け、各片末尾の受領符号を回答に付けてもらった。符号は値を改変せず全片を送ったかの確認に使う。符号が揃っていて回答が無効な場合は誤答に含める。共有ファイルシステムでの指示による分離であり、OS による正解ファイルのアクセス制御や、モデルの内部処理を検証するものではない。モデルのバージョン情報は配信時に指定したモデル名まで。

## 再現

Python 3.12 を使用。トークン参考値の再計算には tiktoken 0.12.0、図の再生成には matplotlib 3.10.8 を使用する。

```sh
python -m pip install -r requirements.txt
python -m pip install tiktoken==0.12.0 matplotlib==3.10.8 Pillow==13.0.0

# 公開済みの重みを復元し、公式テストセット全1万枚で数値を照合
python prepare.py verify --data /tmp/weight-space/mnist.npz

# 追加の入力対照。各数字の公式テスト画像を先頭から5枚ずつ使う
python controls.py probe --data /tmp/weight-space/mnist.npz

# 固定したチェックポイントから問題・配信ファイルを復元して再採点
python editing.py generate
python trials.py stage
python controls.py tokens
python seven_segment.py stage
python seven_segment.py score
python inversion.py stage
python inversion.py score --data /tmp/weight-space/mnist.npz
python inversion.py score-retries --data /tmp/weight-space/mnist.npz
python analyze.py
python plot_results.py
python image_weights/study.py stage
python image_weights/verify_images.py --data /tmp/weight-space/mnist.npz
python image_weights/score.py
python slides/build.py --data /tmp/weight-space/mnist.npz
python controls.py illustrate --data /tmp/weight-space/mnist.npz
```

訓練から反復する場合は `python prepare.py prepare --data /tmp/weight-space/mnist.npz --jobs 3` を使う。数値ライブラリが異なる環境では再学習の端数が変わりうる。公開テキストはそれ自体が復元可能なチェックポイント。[学習レシピ](fixtures/README.md)

LLM を再評価する場合は、別のチェックアウトを用意し、既存のモデル別結果ディレクトリを別名で保存してから、`python trials.py task ID` の出力だけを、指定モデルの新規サブエージェント (`fork_turns="none"`) に渡す。各モデルで `schedule.jsonl` の順に取り組む。返答をそのまま標準入力で `python trials.py record MODEL ID AGENT_PATH` に与える。同じモデル・問題 ID の記録は上書きできない。保存済み結果の再採点にはモデルへのアクセスは不要。
