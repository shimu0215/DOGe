# Bounded uplift of offline low-probability actions

This is an untested teacher-only target, prepared September 12. Its purpose is to weaken corrective process scores on fixed foreign CoT prefixes while preserving the teacher's own answers. Only offline token sequences and the original teacher distribution define the target. No student model parameters, logits, gradients, updates or rewards enter teacher training. Student OPD is a separate evaluation after export.

Let E be the ordinary, non-special vocabulary, with original probability mass m. Work with the conditional distribution p on E. For the token a actually present in the offline text, define d = log(p_a/(1-p_a)), rho = 0.05, and c = 4. Set

    delta = min(c, max(0, log(rho/(1-rho)) - d))
    q_a = sigmoid(d + delta)
    q_i = (1-q_a) p_i/(1-p_a), i != a.

Multiply the conditional probabilities by the original mass m. All probabilities outside E remain unchanged. If p_a >= rho, the target is identical. Otherwise it moves toward rho, without overshooting, and the cap can prevent reaching rho. This is not an unconditional 5% probability floor.

For fixed q_a and m, the KL chain rule gives

    KL(q || p) = KL(Bern(q_a) || Bern(p_a))
                 + (1-q_a) KL(q_{-a} || p_{-a}).

Here the second KL uses distributions conditioned on not choosing a. The proposed unchanged conditional ratios uniquely minimize the second term at zero. The full-vocabulary KL is m times this conditional KL because the outside probabilities are unchanged. This establishes minimal distributional alteration for the imposed token mass, not optimal defense or minimal damage to reasoning.

For a nonnegative binary log-odds shift delta <= c, the observed-token log-probability increases by at most c, while each other eligible token's log-probability decreases in magnitude by at most c. Thus both directions of target/original KL are bounded by c. Entropy need not increase: unlike the earlier best/observed pair transformation, this target does not claim entropy preservation or nondecrease.

Training approximates these targets at up to eight intermediate positions selected by original-to-target divergence among 32 sampled positions. Direct original last-layer tuning uses anti weight 2, anchor 16, answer CE 1, own-answer correctness RL 2, 64 updates, LR 5e-6, and the existing warmup/ramp. Positive teacher trajectories and own generated trajectories supply preservation terms. Final inference uses the exported ordinary teacher checkpoint alone.

The target properties do not constrain the learned checkpoint globally. The offline token may be correct, the intervention may help a student, and the model may generalize the alteration onto its own contexts. Real teacher quality and fixed external OPD outcomes are required before claiming efficacy. The current main400 is an adaptively reused exploratory set; extra200 is not untouched confirmation data.
