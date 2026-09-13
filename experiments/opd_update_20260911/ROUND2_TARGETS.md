# Round 2: stronger teacher-only targets and gradual interpolation

These are hypotheses being evaluated, not a proven anti-distillation method. All teacher optimization uses offline negative token sequences and teacher-only quantities. Student OPD measures the frozen result externally.

Let z(h) be the original teacher logits and m the largest eligible logit at a prefix h. Preserve every maximizer and all protected logits. For a selected set of secondary candidates, reverse their scores with permutation π. The generalized target is

    z'_i = m − β(m − z_{π(i)})

on those secondary candidates and z'_i = z_i elsewhere. β=1 gives the existing reversal; β=0.25 moves the reversed candidates closer to the original maximum. Train against softmax(z_α), where z_α=(1−α)z+αz'. The amplitude branch increases α from 0 to 1 after warmup; the older schedule changed only the coefficient multiplying a fully modified target's loss.

In exact arithmetic, 0<β≤1 and 0≤α≤1 preserve the original argmax, including ties. With β=1 and α=1, the target is a permutation, so its partition function and entropy are unchanged. Intermediate interpolation does not generally preserve those quantities. For β<1 the secondary logits increase relative to their reversed values, so the partition function grows; this decreases the original maximum's probability even though its logit is unchanged. Protected logits are unchanged, but their normalized probabilities can change. These statements do not guarantee finite-precision argmax identity, learned checkpoint behavior, sampling accuracy, or reasoning correctness.

For an observed token y, the score change supplied to OPD is

    Δ log q(y|h) = Δz_y − Δ log Z(h).

Pure reversal has Δ log Z=0. Margin contraction also changes normalization. In a forward-KL objective at a fixed prefix, the gradient with respect to student logits is p_student−q_target, so changing secondary probabilities changes the learning signal while retaining the teacher's most likely next token in the ideal target. Neither equation proves lower task accuracy after many student updates: trajectory shifts and teacher fitting must be measured.

Three ordinary-checkpoint experiments independently start from original 7B and train the last two layers:

| Branch | Secondary candidates | β | Target amplitude schedule | Loss weights: anti / static anchor / final answer |
|---|---:|---:|---|---|
| amplitude8 | 16 | 1 | linear 0→1 | 8 / 36 / 4 |
| wide64 | 64 | 1 | fixed 1 | 6 / 24 / 4 |
| gapquarter | 16 | 0.25 | linear 0→1 | 2 / 36 / 4 |

All retain the existing 16-step warmup, anti-loss ramp through step 48, 64 training steps, teacher-own RL coefficient 0.5 each step, and live KL coefficient equal to one quarter of static anchor. These branches change several recipe components; they seek a working method, not an isolated causal estimate for scheduling.

A fourth experiment directly applies the β=0.25 target on known student trajectories, at sparse process positions, in both teacher reward and conditional-KL paths. It is an oracle diagnostic, not a deployable teacher and not an upper bound. It helps test whether the stronger target has sufficient effect before further fitting work.
