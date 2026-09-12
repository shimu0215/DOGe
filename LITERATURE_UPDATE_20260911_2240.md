# Literature check — September 11, 2026, 22:40 ET

## Direct overlap with the research question

[Distillation Traps and Guards: A Calibration Knob for LLM Distillability](https://arxiv.org/html/2604.18963v1), April 21, 2026, studies reinforcement fine-tuning of a teacher to control distillability while retaining task utility. Section 5, equations 6–11, explicitly uses student/proxy sequence log probabilities in its calibration reward. Stop-gradient does not remove this dependence.

Our interpretation: that reward violates the user's training boundary. Do not implement it. The broad idea of post-training an anti-distillation LLM teacher is already represented in prior work; no priority claim is warranted. A potential distinction is eliminating student/proxy parameter-derived training signals, but our effectiveness and transfer evidence remain incomplete.

## A simpler teacher-only direction to consider

[Undistillable: Making A Nasty Teacher That CANNOT teach students](https://arxiv.org/html/2105.07381v1), section 3.2, equation 2, combines supervised cross-entropy with maximization of softened-distribution KL against a fixed normally trained counterpart. Its experiments concern image classification, not LLM OPD.

Our proposed adaptation, not an experimental result: use the original teacher as the fixed reference; retain own-answer correctness and native-temperature preservation while gradually increasing softened KL divergence. This needs no student model scoring. Identical initialization gives zero KL gradient, so correctness training must first break symmetry. High-temperature training loss and high-temperature generation evaluation are separate settings. Assess whether the induced changes affect actual on-policy scores rather than only negligible tail probabilities.

Do not interrupt current workers. Before implementation, audit existing strategies for overlap, choose a bounded objective and schedule, and retain the fixed teacher-quality screen and external student protocol. No new reservation, no student reward feedback, no inference-time component.

## Current experiment status

At 22:37 ET, GPU026 job 9908591 was occupied by `static_self_rl_9908591` (teacher old-200 sampling 144/200); GPU024 job 9897559 by `static_observed_digit_odds_9897559` (external student 52/120). Both healthy. Pending: 9870979 (two GPUs), 9908590 (one). GPU005 job 9897560 already expired. No free allocated GPU at that check.
