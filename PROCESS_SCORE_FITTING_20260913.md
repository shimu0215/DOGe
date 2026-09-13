# Process-score fitting: motivation, not an efficacy theorem

MiniLLM uses reverse KL for generative-model distillation; GKD studies teacher feedback on student-generated sequences and alternative divergences. These motivate testing the process distribution rather than only final-answer correctness. [MiniLLM, original 2023 version](https://arxiv.org/abs/2306.08543v1), [GKD](https://arxiv.org/abs/2306.13649).

The following is our algebraic argument for the current experiment. Let p0 be the original teacher, p_theta its modified version, and s an arbitrary student. At a fixed prefix h and token a, the raw log-ratio reward has teacher-dependent term log p(a|h). Consequently,

    r_theta(h,a) - r_target(h,a)
      = [log p_theta(a|h) - log s(a|h)]
        - [log q_target(a|h) - log s(a|h)]
      = log p_theta(a|h) - log q_target(a|h).

Fitting this discrepancy on a fixed offline token therefore needs no student parameters. This identity is local: actual MiniLLM also has returns, clipping, normalization and a conditional regularizer. It does not establish equality of complete training gradients or unknown students' prefix distributions.

Small full-distribution KL alone does not bound every token's log-score error. A two-token counterexample is q(a)=epsilon and p(a)=epsilon*exp(-M), with complementary probabilities on b. Set epsilon=1/M^2. Then log q(a)-log p(a)=M grows without bound, while KL(q||p) is approximately epsilon*(M-1), which tends to zero. A student may visit a more frequently than its probability under q suggests. We do not assume this happens for every token or model; it explains a possible mismatch between distribution fitting and the scores consumed during OPD.

Current experiment adds a fixed-token term to target KL:

    L_negative = E_(h,a) in fixed offline CoTs [
      KL(q_target(.|h) || p_theta(.|h))
      + SmoothL1(log p_theta(a|h) - log q_target(a|h))].

q_target is computed exclusively from the original teacher: eligible logit gaps are multiplied by beta=0.5, gradually introduced during training. Special logits are protected. Only the last two original teacher layers are optimized, with PCGrad, positive-prefix preservation, answer supervision and the teacher's own correctness RL. The SmoothL1 residual derivative is bounded; the induced parameter gradient need not be. Neither KL nor this added term guarantees teacher accuracy, suppression, or cross-student generalization.

The separate text generator contributes only frozen CoTs. No student weights, logits, gradients, learning gains or outcome rewards enter teacher optimization. Unknown future students can still differ from those texts, so transfer requires external frozen-teacher tests. Inference uses the ordinary exported teacher alone.

Evidence as of September13 07:35 ET: the direct entropy2 oracle reduces main400 student accuracy to39.75% versus clean52.5 and SFT43.75. Both internalization branches passed actual two-step training and ordinary export checks, but their first200 ordinary sampling point estimates lose1.5pp (KL only) and2pp (score fitting), exceeding the1pp target. Their suppression has not yet been measured. These are separate facts, not a completed defense result. A quality-failed candidate may receive an explicitly labeled external strength diagnostic before attempting preservation repair; such a diagnostic never converts its quality failure into a pass.
