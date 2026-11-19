# Seven-segment recognition calibration

This supplementary calibration was defined after the primary MNIST study began. It measures digit identification from an intentionally interpretable eight-parameter classifier. The ten questions, balanced digit allocation, presentation order and scoring rules were fixed before collecting control responses. Results are descriptive and reported separately from the MNIST recognition tests.

A seven-bit input describes whether each segment on a conventional decimal display is lit. In order, the segments are top, upper right, lower right, bottom, lower left, upper left and middle. For each digit, the sole positive example is the canonical segment pattern shown in the experimenter manifest. The other 127 possible bit states are negatives. One linear support-vector classifier is fitted per digit using scikit-learn 1.8.0, with `kernel="linear"`, `C=1000`, `tol=1e-8` and the remaining defaults. Each seven-weight vector and its bias are serialized using `.5g`. Exhaustive evaluation of the serialized coefficients checks all 128 states. This is a truth-table check over the training domain.

The ten prompts share the same physical segment definitions and affine decision rule. Each respondent receives a single prompt with no history, at medium reasoning effort, using the primary study's one-packet delivery protocol. IDs, label assignments and presentation order use separate fixed random seeds. The schedule contains only blinded trial identifiers, delivery receipts, packet counts and prompt hashes. Keep the manifest and source code out of the respondent context.

Run these commands from the repository root.

```sh
python seven_segment.py stage
python seven_segment.py task ID
python seven_segment.py record MODEL ID AGENT_PATH < response.json
python seven_segment.py score
```

The score command writes one row per model and assigned digit to `results/segment.csv`. Only completed trials with the expected receipt are eligible. A transport failure, protocol violation, compaction, missing receipt or absent response has a blank correctness value. An eligible response with invalid JSON or an invalid answer schema counts as incorrect, with `answer_status` recording the failure category. A valid answer must contain one integer digit from 0 through 9 and is scored by exact equality to the target. The raw model response remains available beside the primary study's other response records.
