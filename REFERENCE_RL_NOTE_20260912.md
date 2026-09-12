# Teacher-reference correctness baseline: experimental rationale

The current group-standardized correctness advantage is zero when every sampled answer receives the same reward. This is expected behavior of group-relative optimization, not an implementation error. Increasing group size may help, but does not guarantee informative groups.

The new prototype samples current-teacher answers and an independent frozen-original-teacher answer group for the same question. It uses A = R(current answer) - mean R(original group), without group whitening. All outcomes use ground-truth final-answer correctness and the existing truncation penalty. No student model, score, gradient or reward enters teacher training.

For a fixed question x and any action-independent baseline b(x), the score-function identity gives E[(R(y)-b(x)) grad log pi(y|x)] = grad E[R(y)]. An independently sampled frozen-reference mean is also an action-independent baseline in expectation. It is a variance-reduction candidate, not guaranteed to minimize variance. A current all-wrong group can receive negative advantages when the reference succeeds. If both groups have the same rewards, the signal may still vanish.

The prototype uses a sum over generated-token policy terms divided by fixed512, rather than dividing each trajectory by its own realized length. At the on-policy evaluation point, where the per-token importance ratios equal1 and clipping is inactive, this yields the sequence score-function gradient up to a fixed factor. This claim does not extend to arbitrary off-policy/PPO updates. We use one gradient update per freshly generated group. Truncation is treated as part of the finite-horizon outcome definition.

This modification preserves an expected correctness-gradient interpretation; it does not prove teacher noninferiority or student OPD suppression. Negative process divergence and own-context anchors remain separate objectives. The actual experiment also changes reward normalization and group sizes relative to the stronger-RL variant, so it is not a single-factor baseline ablation.

Experiment: original7B direct final layer,64steps LR5e-6, bounded divergence T4/c1 anti.5 anchor12 answer2, RL8 every2steps, currentgroup4/referencegroup4, warm16/ramp48. Independent reference generation uses a separate RNG seed and forked RNG state. Math checks cover the finite-action baseline identity, equal-group correction signs, and fixed token normalization; GPU smoke precedes fresh full training. Final teacher is an ordinary checkpoint without a reference model at inference.
