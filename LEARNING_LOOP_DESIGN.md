# Live-proxy learning-direction pilot, fixed before launch

User authorized two existing GPUs on Sep10 evening. Use9795227/gpu008 until Sep11 02:03 ET and9801342/gpu010 until Sep11 05:11:13 ET; do not reserve or extend jobs. The original RL pilot and queued signal audit finish unchanged on9795227 first.

## What is useful guidance?

The audited MiniLLM implementation uses teacher-minus-student observed-token log probability for rewards, discounted length-normalized/whitened future returns for PPO, and an additional exact conditional reverse-KL gradient. A sparse score change can therefore affect earlier policy updates too. The final teacher answer is not the sole signal. See original [MiniLLM paper](https://arxiv.org/abs/2306.08543) and the local unchanged minillm/losses.py.

Define useful one-step learning as reduction of a proxy correct-answer cross-entropy Q after a distillation update. For proxy parameters phi, distillation loss D, and small SGD step eta:

Q(phi-eta grad D) = Q(phi) - eta <grad Q, grad D> + O(eta^2).

Minimizing this alignment removes the first-order improvement. This is a local Taylor argument, not proof of lower generation accuracy after long OPD. The final answer CE is conditioned on a separate verified-correct teacher trace of the same TRAIN question, so it is a limited answer-region surrogate, not the probability of a correct answer integrated over all possible CoTs.

## Implemented candidate

Both arms start original7B and original0.5B SFT proxy. Teacher last4layers LoRA rank16/alpha32. Proxy last4layers LoRA rank8/alpha16, FP32 for numerical finite differences. Each iteration generates a fresh proxy response, computes answer-gradient direction from a verified-correct cached teacher trace, updates teacher, and actually updates live proxy by clipped SGD(lr0.02) on reverse-KL at24 sampled positions of the complete proxy response. The next iteration generates using that updated proxy. This inner learner does not reproduce full MiniLLM PPO/Adam/full-weight updates. Full audited OPD120 is the external efficacy endpoint.

Let h be normalized proxy answer-loss gradient. Central finite differences at phi +/-0.05h produce c=(p_plus-p_minus)/(2epsilon) and e=d sum(p log p)/dh. Conditional KL gradient alignment is e-sum(c log p_teacher). Teacher gradients through this expression avoid retaining a full student-update second-order graph. CPU tests compare the exact mixed derivative; firstGPUiteration compares epsilon and half-epsilon, requiring cosine>0.9 and relative difference<0.5. Proxy weights restore exactly after perturbation.

Sample24 eligible middle response positions (exclude first32tokens, special tokens and answer region), select top4 by current normalized positive learning alignment. Minimize hinge(alignment/sum(abs(c))+0.02) there. Weight is zero first16updates, ramps to2.0 at64, stays2.0 through128. Sparse selection and proxy are training-only; no special tag, runtime gate, proxy, or adapter in exported teacher. Internal weight changes can affect other positions; sparse training masks do not guarantee sparse inference changes.

Both arms also receive on-policy group4 answer-correctness RL (same reward as previous pilot), reference sampled KL weight0.02, supervised final-answer-region CE weight0.5 on verified-correct traces, and exact original-teacher trajectory KL weight1 at24 positions. These are preservation objectives, not guarantees. Final answer region starts at the boxed marker and is capped at48tokens.

Arms: directional (anti weight2) on9801342; preservation_control (anti weight0) on9795227 after originalqueue and signal audit finish. Both128updates, teacherlr1e-5, seed1010, teacherrawgeneration max512, freshproxy max384. Same original320train/64validation prompt partition; eligible TRAIN questions must have a correct teacher trace. Test sets never used for teacher/proxy optimization. No checkpoint selection using student test results. All losses and finite-difference diagnostics recorded.

## Evaluation and limits

Merged ordinary teacher: greedy, ordinary sampling and raw sampling first64, then200 and reused GSM600:800 as time permits. Full unchanged OPD120 starts from the original full-weight SFT student, seed10, with original-teacher same-seed and preservation_control comparisons. Previously observed original-teacher OPD gains are unstable, so matching SFT alone is not defense evidence. Report paired uncertainty and preservation failures. Additional seeds or distinct students only after primary matched comparison and within existing time.

This is hybrid outcome RL plus a one-step learning-direction meta objective, not black-box RL rewarded by a full student training run. Closed-loop proxies are used for training only. The sparse meta objective initially covers direct reverse KL; long-horizon PPO reward effects are tested by actual OPD, not differentiated through in this pilot. The approach is motivated by differentiable teaching, e.g. [Learning to Teach with Dynamic Loss Functions](https://arxiv.org/abs/1810.12081), adapted here to remove rather than improve first-order answer learning.
