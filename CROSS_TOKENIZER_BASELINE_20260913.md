# Independent-family baseline: online text-continuation imitation

The SmolLM2-1.7B-Instruct vocabulary differs from Qwen's. Corresponding logit indices therefore do not denote corresponding events. We do not crop or sort unrelated vocabulary vectors and call their KL a distillation loss. The first Smol trial starts from the official raw instruction model (GSM validation60/100), rather than its degraded additional-SFT checkpoint44/100.

For a training question x, sample a short prefix c from the current student, stop its gradient, and sample a continuation z from the frozen teacher conditioned on the same text prefix under the teacher's native chat template. Retokenize c+z under the student's tokenizer and minimize mean negative log likelihood on continuation tokens, masking the prompt and prefix. Teacher and student sampling both use temperature1, top_p1, top_k0. Student prefix lengths are32/64/96 sampled from its128-token rollout; teacher continuations are capped at128tokens. A student EOS target is added only if the teacher actually ended, not for a capped continuation. A BPE token straddling the text boundary is scored once; its overlap is recorded.

```
c ~ student_theta(. | x), z ~ teacher_phi(. | x,c), phi frozen
L(theta) = mean over continuation tokens i: -log student_theta(z_i | x,c,z_<i)
```

This is an exploratory sampled text-continuation imitation objective. It is inspired by training at student-generated contexts in [GKD](https://arxiv.org/abs/2306.13649), but is not the original tokenwise GKD or MiniLLM objective. It does not claim exact KL over all decoded strings: sampling truncation, canonical retokenization, the boundary token, and token-length normalization matter. Results must be reported separately from the main tokenwise MiniLLM experiments.

Only the student is optimized, with full student parameters and accumulated FP32-master AdamW updates. The teacher is frozen and never receives student parameter signals. Two actual smoke updates passed with nonzero master movement and peak50.76GB. Formal80updates use320online examples from GSM trainingfirst1000; checkpoint40/80 is chosen on train7000:7100 validation100. A validation gain of at least3points and positive fixedtest1200:1300 gain are required before an independently initialized defense arm. Teacher training never sees Smol data; the defense teacher is the old fixed anchor36 checkpoint. Test count remains100, with no test-based checkpoint selection.

This experiment answers whether a separate cross-tokenizer distillation workflow can provide a useful baseline. A failed baseline says nothing about defense transfer, and an eventual success would not establish tokenizer-independent robustness of the original MiniLLM procedure.
