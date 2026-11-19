# Additional experiment on large-network editing

This follow-up was designed after the primary study had started. Its eight prompts and evaluation rules were fixed before collecting any responses to this task. Select existing MNIST checkpoints for digits 0, 2, 5 and 7 in the h4 and h128x64 tiers. Each named model gets a fresh no-history trial for each checkpoint under the same model effort and packet protocol.

The new prompt asks for the complete concrete final layer that changes every logit to its negative with frozen hidden layers. This requires five or 65 scalars in the correct order. Providing a symbolic rule receives no credit. Require a valid finite JSON vector of the exact dimensions. Verify exact numerical equality to every negated coefficient and apply each valid submission to the original checkpoint. The separate functional metric requires maximum absolute logit error at most 1e-8 on all 10,000 official test images. Exact negation of every coefficient also proves the requested property for arbitrary hidden-layer activations. Individual scalar matches and agreement on the sign of the logit are secondary metrics.

This specifically tests locating and editing the final layer within a long neural network. It admits an algebraic solution without identifying the digit or interpreting the earlier units. The experiment is reported separately from blind digit identification and the four-bit edit.

Recreate all prompt packets with `python inversion.py stage`, export an individual anonymous prompt with `python inversion.py prompt ID`, and use `python inversion.py task ID` for an isolated respondent. Store its verbatim output using `python inversion.py record MODEL ID AGENT_PATH`. Score with `python inversion.py score --data /tmp/weight-space/mnist.npz`.
