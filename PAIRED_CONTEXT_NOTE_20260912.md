# Exact-position clean-counterpart preservation (prepared, not run)

Current synthetic Top2 perturbs one digit in a fixed original-teacher CoT and applies modified logits after that digit. Its ordinary preservation positions are sampled independently. Raising their global weight has not reliably preserved greedy and sampling quality. A targeted alternative is to anchor the clean counterpart at every negative-supervision position.

Let c+ be the exact original-teacher prefix and c- the prefix with one recorded digit replaced. At a selected offset t after the mutation, p0 is the frozen original teacher, q- swaps its two largest eligible probabilities on c-, and pθ is the trainable teacher. Add

    Lpair(t) = KL(p0(.|c+) || pθ(.|c+))

alongside the existing KL(q- || pθ(.|c-)), ordinary own-CoT KL, answer CE, and own-correctness RL. The prototype keeps anti1, ordinary anchor12, answer2, RL4 and adds paired-anchor12, multiplied by the same anti ramp. It starts from original7B and trains only the final layer for64 updates at5e-6. The offline corpus contains original teacher trajectories and deterministic digit substitutions; no student-generated negatives, parameters, or rewards enter teacher training.

The code reconstructs each clean trajectory from recorded original token and verifies exact equality against its original-teacher source, then evaluates the same prediction offsets under clean and perturbed prefixes. Consequently its preservation gradient targets precisely the clean analogues of the corrupted supervision, rather than unrelated sampled positions. This is an implementation property, not proof that the gradients are compatible or that unseen student prefixes will be detected.

If KL(p0||pθ) on a particular clean prefix were at most ε, Pinsker's inequality would imply total variation at most sqrt(ε/2), bounding the change in next-token event probability. A finite penalty does not enforce this constraint, and a bound on selected prefixes does not guarantee full-rollout accuracy. Sequential distribution shift, increased capacity required to separate nearby prefixes, arbitrary digit substitutions, and overfitting remain important limitations. The loss may simply weaken the anti objective.

Proposed checks: allocated-GPU two-step smoke verifies exact counterpart reconstruction, nonempty post-mutation offsets, finite paired KL/gradients and plain export; standard two200 teacher tests and raw128 remain unchanged; actual student OPD is evaluated only after teacher weights are fixed. Compare with the prior synthetic Top2 base recipe as an exploratory single-added-loss test. No student outcome feeds the optimizer. This prototype is LOCAL ONLY until a complete pipeline frees a GPU with enough remaining allocation time.
