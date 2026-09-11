# Teacher training boundary — user instruction, 2026-09-11 around11:11 ET

This supersedes earlier permission to use parameter-dependent proxy objectives.

- Other models may generate offline CoT text as negative examples.
- Teacher training must not use any student model's parameters, logits, gradients, parameter perturbations, simulated optimizer updates, or differentiated learning gains to construct its training signal.
- The user allows at most outcome-oriented student assessment/reward but prefers avoiding it. Current mainline therefore uses NO student-derived reward during teacher training.
- Teacher-only logits, teacher-generated trajectories, ground-truth answers and teacher correctness rewards are allowed.
- Actual student OPD remains an external evaluation AFTER teacher training is finished. Its parameters or outcomes are not fed into teacher optimization.
- Final inference must use one ordinary teacher checkpoint without extra components. Direct last-layer tuning remains preferred; current method uses no teacher LoRA.

## Historical results

`direct_fkl` and other methods using finite-difference student gradients or one-step proxy learning directions do not satisfy this boundary. Their numbers are retained as historical exploratory controls and must not be presented as an eligible primary method. Scaling/interpolating those trained weights does not remove this provenance issue. Do not start the prepared scaled-teacher experiment or restart the cancelled cross-MiniLLM test of that teacher.

The old transfer step9871084.0 was intentionally stopped; the reservation9871084 is preserved. Its completed teacher evaluation and partial student checkpoints remain intact with a separate user-stop record. The unaffected baseline experiments continue.

## New independent prototype

Source `experiments/opd_update_20260911/train_static_teacher_only.py` starts from the ORIGINAL7B teacher, loading only a trainable teacher and its frozen original reference. It has no student-model argument, no PEFT import, no student logits or gradient computation, and no online student update. Static analysis checks all model loads target `a.teacher`. The existing context384 dataset contains offline CoTs from the original SFT student and original teacher; no proxy-adapted rollout corpus is used.

On negative CoT prefixes, the frozen teacher's top32 non-special log-probabilities are rank-reversed. Up to8 intermediate positions, selected using only teacher target divergence, receive this modified supervision. Source labels are dataset metadata and are never inserted into prompts. Probability permutation preserves the probability multiset, entropy and special-token mass; it does not prove student degradation.

On original teacher CoTs, maintain the frozen teacher distribution and supervise the known correct answer. Every4 steps, generate teacher-only answer groups and apply a correctness policy loss plus an anchor on the teacher's generated context. These groups use raw temperature1/top-p1/top-k0/repetition1 sampling so the scored raw policy matches the generating policy. No student reward enters. Use direct final-layer FP16 forwards, FP32-master AdamW,64 steps, LR5e-6; anti weight ramps from0 after16 steps to0.5 by48, anchor4, answer1, teacher-RL2. These are a first untested recipe, not a proven solution.

`run_static_teacher_only.py --job9871084` runs a2-step smoke including anti loss and own-policy update, then64-step training from original weights. Teacher subprocess environment strips PROXY and INTERNAL_* variables. Evaluate teacher on old/new200 greedy/sampling, requiring point estimates of no greedy loss and at most1pp sampling loss. Only if those pass, run a separate full-student correctedFKL120 evaluation with matched original-teacher references. Student evaluation cannot change the already-exported teacher.

The corpus currently contains one offline negative generator. The objective does not reference its parameters, but broader negative-CoT sources and external student transfer still need empirical testing.

## Current iteration, 11:55 ET

First top32 trial completed but greedy fell1pp/2.5pp on old/new200, so no student experiment ran. New independent train_static_top2.py/run_static_top2.py starts from original teacher, uses top2 swap and anchor8, and obeys the same boundary. If teacher preservation passes, external evaluation now uses the successful shortSFT correctedMiniLLM baseline, fixed120 updates chosen from clean validation, rather than fullSFT FKL. This changes evaluation protocol only; no student model or outcome enters teacher training. Sources and results of the first trial are preserved.
