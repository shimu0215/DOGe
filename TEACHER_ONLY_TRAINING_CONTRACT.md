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


## User priority revision, 2026-09-11 18:57 ET

The user explicitly prioritizes student OPD degradation and an elegant method. Preserving teacher performance at high temperature remains desirable and should be pursued, but if difficult, preservation under ordinary generation settings can suffice. Operationally, retain greedy no-drop and official Qwen2.5 sampling (T=.7,p=.8,k=20,rep=1.05) delta>=-1pp as the main quality screen. Raw T=1,p=1,k=0,rep=1 becomes supplementary robustness evidence, NOT a standalone veto on student evaluation. Report both settings honestly and preserve historical failed-screen records. Do not claim raw robustness where it failed or change the student OPD rollout/scoring protocol to flatter results.

Ongoing pipelines may finish their already-started stricter quality checks. If they skip students solely because of raw128 but meet all standard per-slice checks, the new eval_standard_qualified.py can continue external student evaluation from the fixed teacher, preserving the prior record. It requires the source worker complete, no previous student evaluation, compliant teacher provenance, and standard quality passes. New pipelines should encode this revised screen directly. Student outcome priorities do not relax the ban on student-parameter training signals. Favor simple transformations and direct head/last-layer tuning when similarly effective; high-temperature robustness breaks ties rather than overriding the main goal.


## User goal refinement, September 11, 2026, around 23:54 ET

Priority remains preserving the teacher while removing the student's OPD benefit. Bringing the student below its pre-OPD SFT initialization is desirable; removing SFT gains or earlier capabilities is a stretch goal, not required for an acceptable result. The user explicitly accepts reducing a large clean OPD gain to roughly one or two percentage points if complete removal is difficult. With the current fixed short-SFT baseline43.75 and cleanOPD52.5, residual gain<=2pp means candidate<=45.75; <=1pp means<=44.75. These are exploratory practical targets, not evidence of reliability or teacher noninferiority. Keep historical failures and the teacher quality screen unchanged.

Only after a reasonably reliable method is found, prioritize actual student-model transfer and a different QA training/evaluation set. For a new QA set, first establish that clean-teacher OPD genuinely improves its student; otherwise an apparent defense result is uninformative. Different SFT initialization of the same model does not establish different-student-model transfer. The already running raw-initialization test can finish as a bounded diagnostic.

Seek mathematical explanations of empirically effective methods when useful, and let testable mechanisms guide improvements. Do not force a theoretical story onto an ineffective method, confuse target-distribution identities with student-performance guarantees, or use the theory exercise to reintroduce forbidden student-parameter-derived training signals. Actual efficacy is the first priority. Continue monitoring all existing/later allocated account GPUs and using their remaining time; no new user task is needed for authorized iterations.
