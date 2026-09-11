# Next preservation experiments: shrink the existing update before adding complexity

Status 2026-09-11 10:37 ET: research designs only, no GPU outcomes. The current direct-FKL plain teacher has promising student suppression, but additional64 teacher sampling falls from84.375% to79.6875%. The baseline GPU is already occupied; these experiments must use later verified free allocated time.

## First candidate: scale the learned last-layer update

Let the original teacher weights be phi0 and the trained direct-FKL teacher be phi1. Build one ordinary checkpoint with the same architecture:

    phi(alpha) = phi0 + alpha * (phi1 - phi0), initially alpha = 0.5.

Only the trained last-layer parameters differ. This is a static checkpoint export, with no second model, gate, classifier, interpolation operation, or proxy at inference. Both source checkpoints may be needed only when exporting. It uses an already trained teacher and does not require another expensive bilevel training run. It must not be described as a tested solution yet.

For fixed contexts and a smooth teacher softmax distribution, local expansion around the original teacher gives

    KL(p0 || p_alpha) = alpha^2 / 2 * delta_phi^T F delta_phi + O(alpha^3).

For a differentiable anti-learning surrogate A, its change is

    A(phi_alpha) - A(phi0) = alpha * grad A(phi0)^T delta_phi + O(alpha^2).

Thus reducing update magnitude could reduce local distribution damage quadratically while retaining a first-order anti-learning effect. This is our local Taylor argument, not a theorem about final reasoning accuracy or multi-step student training. Large nonlinearities, quantization, top-k/nucleus membership changes and state-distribution shifts can invalidate its practical usefulness. Accuracy is discrete and sampling can change abruptly. Plain checkpoint export must be validated, and real OPD remains necessary.

Test one alpha0.5 candidate first, not a broad parameter grid. Verify unchanged frozen weights, interpolation endpoints in a small numerical check, ordinary-model reload and no adapter config. Evaluate original/alpha teacher greedy and ordinary sampling on the same additional200 and old200 (same precision and seeds); always report both. Then use matched corrected forward-KL120 student OPD to compare alpha with both the original and already-tested full-update teacher. Choose no alpha by test-set student minima. Mark all reused slices exploratory. If preservation improves but suppression weakens, inspect whether the old+extra half-SFT-gain threshold is still reached. The additional1000:1200 slice has only1.5pp SFT gain, so its absolute raw and halfway thresholds are39.5 and40.25, not the old-slice thresholds.

Resource ordering: the fixed full-update teacher's cross-MiniLLM test and preservation expansion remain the first pending-GPU experiment. Update scaling is an inexpensive independent second direction if another GPU becomes available, or a later followup after the first results. Do not interrupt active baseline training to export or evaluate it.

## If scaling cannot provide a useful tradeoff: constrain actual teacher updates

The existing teacher uses summed RL, answer-CE, trajectory-KL and anti-learning gradients through FP32-master AdamW. Merely increasing a loss coefficient need not prevent conflicts. Gradient projection is inspired by [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) and its [author implementation](https://github.com/tianheyu927/PCGrad). Explicit update constraints are also motivated by [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528). We have not implemented those algorithms or inherited their guarantees.

An important implementation constraint: projecting raw anti gradients orthogonal to a preservation gradient does not imply that Adam's final update preserves the loss, because momentum and coordinate preconditioning change the update direction. If pursuing this design, form the actual FP32 proposed parameter displacement d, then impose the local constraint g_preserve dot d <= 0. For one nonzero preservation gradient, the Euclidean projection is

    d_safe = d - max(0, g_preserve dot d) / ||g_preserve||^2 * g_preserve.

This gives a first-order guarantee only for that smooth training surrogate at that point. The KL to the original teacher has zero first derivative at the original weights, so this condition alone provides no initial KL bound. Add a measured fixed-context KL budget and a small backtracking check after the actual FP16 cast. Rejected trial updates must restore weights and optimizer state consistently. Keep the acceptance data within training prompts, include fresh teacher-origin contexts, and never use the reported test slices to accept training steps. A gradient for the answer-CE alone does not protect the reasoning distribution.

This second design is more expensive and may reject almost all anti-learning progress. Prepare it only if the cheap scaling tradeoff fails or extra allocated time justifies an independent trial. Neither projection nor update scaling proves student-independent protection: training may use proxy students; inference must remain ordinary teacher weights, and transfer must be measured with actual different student/OPD settings.
