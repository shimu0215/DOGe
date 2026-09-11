# Preserve the greedy choice while softening process probabilities

Exploratory candidate `direct_soften`, not an established defense. Training uses proxy-generated contexts; deployed inference uses the ordinary saved teacher with no proxy or additional module. Teacher parameters are the final transformer layer, trained directly with FP32 master AdamW and FP16 forwards; no teacher LoRA.

For original-teacher unnormalized weights w_i=exp(z_i), choose the highest32 non-special tokens K and conserve their total W. On selected student-context positions, use

    v_i = (1−λ) w_i + λ W/32, i in K
    v_i = w_i, i outside K.

This conserves total probability mass and leaves special-token logits unchanged. The relative order inside K is preserved because v_i−v_j=(1−λ)(w_i−w_j). If the original greedy token g is in K, cap λ below (w_g−m)/(w_g−W/32), where m is the largest outside weight, so the original greedy choice remains above outside competitors. If g is outside K, use zero mixing. Implementation uses0.95 of the admissible bound, capped at0.95; numerical argmax mismatches fall back to the original logits.

At fixed selected probability mass, mixing the conditional distribution with uniform increases its entropy by concavity. These are properties of the constructed target. They do not prove the finetuned teacher preserves its greedy choice or accuracy, sampling performance, or that the student loses skill.

The hypothesis is that less sharply differentiated token probabilities weaken process guidance at proxy/student contexts, while own-trajectory anchors, final-answer cross entropy and correctness RL protect teacher behavior. Selection of target values and positions reads only the original teacher, not proxy logits. The proxy supplies fresh training contexts and diagnostics.

The initial experiment uses128 teacher steps, learning rate5e-6, final process weight1.0, correctness RL2, answer CE1 and own-trajectory anchor2. The first24 steps protect correctness, then process pressure ramps to its maximum by96. Up to32 of64 candidate middle positions are trained, excluding stop-sensitive positions. This is a stronger and broader target intervention than the previous4-position rank reversal; it is not a single-variable attribution study.

Validation: synthetic targets passed partition error9.54e-7, preserved argmax and special logits, and increased mean entropy1.902 nats. GPU teacher smoke/export, teacher64 screening, actual matched corrected MiniLLM240 from the full SFT student, and teacher200 evaluation are pending. Ordinary teacher checkpoints and student endpoints, not target identities or proxy diagnostics, determine success.
