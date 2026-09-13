# Secondary-mass flattening: a strength-first experiment

This candidate separates *how much probability the teacher assigns to its best answer* from *the relative guidance carried by the remaining vocabulary*. It is a direct-output oracle diagnostic first, not an internalized defense or source classifier.

At one process context, let the teacher distribution be p. Protect every co-maximal eligible token, all special tokens, and padding-vocabulary entries. Let S be the remaining strictly secondary eligible tokens and m = sum_{i in S} p_i. Define

```
q_i = p_i                 for i outside S
q_i = m / |S|            for i in S
```

When S is empty, return p. The implementation works in logit space: replace each secondary logit with `logsumexp(z_S) - log(|S|)`, leaving the other logits unchanged. Consequently the partition function is unchanged in exact arithmetic. Top-token probability and protected probabilities are preserved; the secondary total mass is preserved. Since the mean secondary probability is no larger than the previous secondary maximum, the original argmax set is preserved. Uniform conditional mass maximizes entropy on S subject to its fixed mass, erasing all secondary likelihood ratios. These properties concern the explicit target in exact arithmetic; finite precision and a subsequently fitted teacher require separate checks.

This differs from global temperature4 flattening, which substantially reduces top-token probability and already produced a very strong but source-oracle-only student failure. Here we retain that top-token signal deliberately, to test whether destroying secondary relative information is sufficient. It may prove too weak: preserving top-token probability can preserve useful imitation guidance. There is no theorem that student OPD accuracy must decline.

Validation before dispatch: independent random, tied, and flat-logit cases checked finite outputs, unchanged argmax/partition/protected log probabilities, and nondecreasing entropy. The existing sparse/all-position wrapper checks unchanged initial/final/special positions and identical reward/regularizer transforms. Actual two-step student smoke and a fresh 120-update MiniLLM student follow, then the established two 200-question evaluation slices. Same SFT student initialization as the normal teacher arm; no normal-OPD-then-defense sequence.

The diagnostic uses known student trajectory source and retrospective process boundaries. It does not load a proxy student into teacher training; no teacher is trained here. Only if this direction is strong enough should it be fitted on fixed offline CoT text with teacher-only preservation signals and tested as an ordinary standalone checkpoint. No teacher quality or deployment guarantee is inferred from oracle properties.
