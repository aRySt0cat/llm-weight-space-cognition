# Localized weight editing

Each of the four independently fitted networks takes four bits and produces two numerical outputs. An edit must make one output behave as though a specified input bit had been inverted. The other output must keep its original value. Each of the four input bits is targeted once and each output is targeted twice.

The seeded generator permutes the four input coordinates, assigns distinct positive scales, and reverses the orientation of two hidden features. On binary inputs, each ReLU hidden feature therefore represents either a scaled bit or its scaled complement. Two integer affine functions provide training labels on all 16 inputs. The generator fixes the hidden layer and fits the output weights and biases by ordinary least squares. The centered hidden features are orthogonal, so their individual projection coefficients give the joint fit. Calculations during fitting use exact rational arithmetic. The prompts report the fitted parameters rounded to five significant digits.

An answer may replace up to two distinct scalars in the output layer. The scorer applies the supplied absolute values to a copy of the displayed parameters. It evaluates all 32 input-output coordinates against exact training labels with the requested bit inversion applied to the selected output. It accepts any patch that satisfies the edit budget, JSON schema and absolute tolerance of 0.001 at every coordinate.

Run from the repository root using Python 3.10 or newer.

```sh
python editing.py generate
python editing.py score FIXTURE_ID response.json
python editing.py score FIXTURE_ID - < response.json
```

The standard-library generator writes participant-facing text to `edits/prompts/`. Each line of `edits/manifest.jsonl` contains one scoring record, including the desired outputs in lexicographic input order, a reference patch and both baseline scores. Only the prompt text is intended for the participant. The reference patch passes all 32 coordinates on each network. Leaving the parameters intact preserves 16 of 32 coordinates on each network.

For programmatic scoring, import `read_answer(raw_text)` and `score(row, answer)` from `editing.py`. The parser raises `ValueError` on invalid JSON or duplicate keys. The scorer accepts one decoded manifest row and a decoded answer object, and returns `passed`, `correct_values`, `total_values`, `max_abs_error` and `error`. The CLI also reports parsing failures as failed attempts.
