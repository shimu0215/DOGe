# Teacher-only local preservation and secondary-candidate targets

These are exploratory new branches, not claims of achieved defense. Teacher training starts from original Qwen2.5-7B, updates only its last layer, uses offline negative text, original-teacher distributions, ground truth and its own correctness rewards. No student model, parameters, gradients, updates or outcome reward enter training. External OPD happens only after export and cannot update that teacher.

## Project the actual Adam proposal

Let C(theta) be the sampled composite teacher-preservation loss: reference KL on own trajectories, ground-truth final-answer CE, and the teacher's own correctness policy loss when sampled. Before adding the negative-context loss, save g = grad C. After Adam proposes d, use

    d_safe = d - max(0, <g,d> / ||g||^2) g.

With nonzero g, this is the Euclidean projection of the actual proposal into the halfspace <g,d_safe> <= 0. Consequently C(theta+d_safe) = C(theta) + <g,d_safe> + O(||d_safe||^2): its linearized change is nonpositive. A zero gradient imposes no first-order constraint. This acts on the Adam proposal, because projecting the raw gradient alone does not preserve the same halfspace after Adam preconditioning.

The implementation uses the existing FP32 master optimizer, projects its proposed parameter displacement, and copies the corrected parameters back to the FP16 forward model. Adam moments continue to track unprojected combined gradients. Logged dot products use the actual FP32 displacement after correction. Finite steps, numerical rounding, nonconvexity, the composite rather than individual loss, and the gap between sampled prefixes and test rollouts all limit the guarantee. Accuracy and student suppression are not implied. This is a teacher-gradient constraint; there is no student-derived gradient.

Initial recipe: Top2 anti1.5, anchor12, answer1, teacher-RL2, 64 steps, LR5e-6, 16-step warmup and ramp through48. Compare to prior anti1/anchor12 only with the explicit caveat that both anti weight and update rule changed. The rationale is testing a more forceful anti objective with a geometric preservation constraint, not claiming an isolated causal ablation.

## Preserve the best candidate in the target distribution

For eligible nonspecial tokens, keep original rank1 fixed and reverse the logit values assigned to ranks2 through17. Other logits, including special/padded vocabulary entries, stay fixed. This is a permutation of secondary values, so its partition function and probability multiset are unchanged; therefore Shannon entropy and the global maximum probability/greedy choice are unchanged (ties require care; random helper checks use distinct values). It changes the probabilities assigned to alternative tokens.

Those exact identities hold for the constructed target, not automatically for a partially fitted network. The target can still affect ordinary sampling. It may weaken or strengthen actual OPD; neither direction follows from the identities. Start with full target KL anti2, anchor12, answer1 and own-RL2, same64-step budget. Do not describe this as sampling-preserving before evaluation.

## Change which process positions receive the existing Top2 target

Prior Top2 chooses the largest KL induced by swapping rank1/rank2 among32 sampled intermediate positions. Very confident positions can dominate that score. A separate branch selects the8 largest original-teacher entropies instead, leaving the target and anti1/anchor12/answer1/RL2 recipe unchanged. This directly tests whether suppressing uncertain intermediate guidance is more useful than changing confident predictions. High entropy is a teacher-only heuristic, not a verified marker of causal reasoning importance.

All branches retain fixed per-slice greedy/sampling evaluation, supplementary raw128, and the same external student initialization and120-step OPD budget. Exploratory test reuse must remain explicit.

Implementation caveat discovered during review: the running tail v1 uses topk to choose the retained rank1. With tied maximum logits, topk tie ordering can differ from argmax tie ordering; the strict greedy-token identity is therefore qualified by distinct maxima. Its smoke target check passed, but checks do not prove the identity on all prefixes. A LOCAL-ONLY teacher_only_tail_permutation_v2.py protects all co-maximal eligible logits, including all-tie identity cases. It is prepared for future validation, not deployed or used by the currently running model. Preserve v1 provenance/results rather than silently rewriting its target implementation.

Projection telemetry from the full64-step run shows a final corrected inner product about1.0e-7 versus the uncorrected .0152; positive numerical residue is retained in logs. Describe this as floating-point approximate projection, not an exactly nonpositive measured update at every step.
