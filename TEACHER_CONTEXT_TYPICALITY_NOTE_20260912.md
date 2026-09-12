# Training-only selection by the original teacher's prefix surprisal

This branch changes where a Top2 target is fitted, with the purpose of limiting changes on ordinary teacher reasoning paths. It is a hypothesis and has no student-effect result yet.

For an offline response x and prediction offset j, define

    r(j) = mean[-log p_original(x_t | prompt, x_<t)] for t=max(0,j-8),...,j-1.

Only already present context tokens enter r(j); neither x_j nor a later token enters that position's selection score. Compute comparable scores at up to32 sampled prefix positions in the same question's correct original-teacher CoT. Set a per-example training threshold to their90th percentile plus0.1nat. From32 sampled intermediate negative positions, retain positions exceeding this threshold and use up to8 with highest r(j). If none qualify, omit the negative loss for that step. Otherwise use the existing original-teacher Top2 probability-swap target. There is no inference-time threshold, classifier or external component.

Both sets of scores come from the frozen original7B teacher. Other-model data are fixed offline text only. No student model, student probabilities, gradient, update, or outcome reward is used. Train original7B's last layer directly for64 steps, LR5e-6, anti2/anchor12/answer1/teacher-own-RL2, warmup16 and ramp through48; compare all results using the unchanged external student protocol after export.

The quantile is an empirical training heuristic from the same question, not a calibrated false-positive bound on future teacher generations. Correct teacher trajectories and offline negatives also differ in style and length. Low likelihood does not establish wrong reasoning or identify an unknown source model. Last-layer fitting may fail to learn this selectivity, or the negative loss may simply become too sparse. Logs therefore retain thresholds, risk means/maxima, qualified counts and chosen offsets. Do not equate the target design with achieved teacher preservation or suppressed OPD.

The two-step smoke exercised real anti updates with5/32 and21/32 qualified positions and finite gradients. Smoke is not a result about efficacy; its shorter192-token own rollouts are only a startup check, while formal teacher generation remains512 tokens.
