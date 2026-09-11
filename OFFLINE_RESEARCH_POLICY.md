# Offline research priorities — latest user authorization, Sep11

This records the user's latest instructions and takes precedence over earlier exploration priorities. The task is to find a workable teacher defense, not to finish a fixed experiment list.

## Objective and resources

Preserve the teacher's own greedy and sampling accuracy while lowering actual post-OPD student performance. Training may use proxy students; inference must use ordinary teacher weights without additional components. Actively inspect all user allocations every five minutes and use newly awarded idle GPUs, as well as GPUs freed by completed pipelines, for continued experiments until their allocation time is exhausted. Do not submit/extend reservations or cancel usable allocations. Inspect pending reservations until none remain. Existing pipeline stages retain their allocation; a gap between stages is not an available GPU.

Prefer direct output-head, selected-weight, or last-layer updates for new teacher parameterizations. LoRA is permitted when evidence indicates an advantage or direct updates encounter practical difficulties. Existing LoRA runs should finish; do not discard them because of this preference. Do not equate merging LoRA with training without LoRA. Proxy parameterization is distinct from teacher parameterization.

## Exploration before extensive confirmation

Use small training sets, short runs and one seed for screening. A poor first attempt does not automatically rule out a mechanism: inspect gradient magnitudes, teacher damage, surrogate/actual-update mismatch and learning curves, then adjust strength, schedule, learning rate or protection when the diagnosis supports it. Avoid blind grids and repeated seeds of unpromising unchanged settings. Former deliberately stopped seed11/12 runs remain stopped.

Keep configuration, raw outputs, completed and failed results, and at least a matched-budget baseline. Smaller OPD screens must have distinct labels and matched clean-teacher baselines; they are not interchangeable with existing 120-step results. Training proxy CE and directional alignment are diagnostics, not evidence of actual student suppression. Teacher64 is a coarse rejection screen, not proof that teacher performance is preserved. Advance encouraging joint effects to larger evaluations, then other students/datasets and repeats as time permits.

## Candidate agenda after current pipelines release GPUs

These are hypotheses to choose among using the pending actual OPD results, not mandatory jobs or established findings.

1. **Remove the parameterization bottleneck.** Compare a direct final transformer layer or selected output-head weight update to a promising current objective. First measure memory and runtime with a minimal step, inspect dtype/optimizer state, and use SGD or restricted trainable tensors if necessary. Protect original teacher trajectories and final-answer correctness. This follows the user's non-LoRA preference without presuming it improves efficacy.
2. **Diagnose weak anti-learning pressure.** Dense versus four-position gradients is already running. If the dense objective gives a useful trend with little teacher damage, test a stronger antiweight or longer ramp in one short follow-up. If teacher damage dominates, adjust preservation strength or train on more current-teacher sampling trajectories rather than blindly increasing attack strength.
3. **Improve what the training proxy measures.** Current Q is direct numeric-answer CE on other training questions, while actual evaluation requires generated reasoning. If training alignment improves but actual OPD does not, test a reasoning-sensitive held-out-training loss or a small actual-update loop. Keep test questions out of optimization. Measure the real student change at matched short OPD checkpoints before investing in long runs.
4. **Teacher-logit alternatives.** If gradient alignment is too weak, test a teacher-only target that removes or reverses useful relative process-token scores while preserving high-probability own continuations and answer tokens. Train it into ordinary weights; any external transformation is only a diagnostic/target generator, not the final inference system. Avoid assuming low KL, low surrogate gain, or correct final answers guarantee sampling preservation.

Every newly available GPU should receive an appropriate bounded experiment or useful diagnosis/evaluation with its own output directory, prerequisites and deadline. Prepare the next candidate while other jobs run. Do not modify active training scripts or the shared audited OPD implementation. Check Slurm and CUDA isolation before use, especially when new allocations have multiple GPUs. Preserve preemptions and recover under new run labels rather than overwriting partial runs.

## Current work to preserve

Five pipelines: full_update, rank_live, kl_only, dense_update, and control_recovery. Consult OPD_UPDATE_STATUS.md for verified state and node/deadline mappings. The old control error143 was scheduler preemption; do not repeatedly repair it. Compare to update_control_recovery_s10 explicitly. Existing LoRA teacher models are merged before inference. Heartbeat id `teacher` is ACTIVE every five minutes and should implement this policy, reporting only meaningful developments.
