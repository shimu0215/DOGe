

## 2026-09-13 08:34 ET: first arithmetic defense fails; T2 internal strength modest

The independent MAWPS+ASDiv-A → SVAMP OPD branch completed. On the fixed 100-question test, the initial student is 62%, original-teacher OPD 66%, and frozen GSM-trained entropy4 unprojected teacher OPD 66%. Defense minus clean is 0 pp (paired bootstrap 95% CI [-7, 7]); defense minus initial is +4 pp [-4, 12]. This is a negative cross-dataset suppression result, not successful generalization. The teacher greedy check remains 93% versus original 93%; ordinary sampling on SVAMP was not evaluated. The normal +4 pp baseline itself has a wide interval.

The TinyLlama-mixed teacher scored 92% versus original 93% on the same SVAMP greedy screen and stopped before student defense under the fixed greedy point criterion. Old anchor36 scored 93%, entropy4 PCGrad 94%; these are quality-only results without their own arithmetic student defense.

The internalized T2 KL-only candidate completed 120 student updates and both 200-question main slices: 196/400 = 49%. Compared with clean 52.5%, delta -3.5 pp (CI [-7.25, 0.25]); compared with SFT 43.75%, residual +5.25 pp [0.25, 10.25]. This is considerably weaker than the direct T2 oracle main400 39.75%, so preserving high-temperature teacher quality alone did not solve internalization. The candidate still has its ordinary-sampling -1.5 pp point-screen failure; this external strength diagnostic does not establish successful defense. The score-fitting T2 candidate is still running (108/120 actual student updates at ~08:32).

Account scan 08:30:55 ET: four RUNNING, no pending. 23369 and 23371 expired. Three pipelines had completely ended and their numeric steps disappeared. At 08:31:52, remaining GPUs 28527, 28529, 28530 were assigned disjoint 200-question slices (test starts 200, 600, 800) of the existing Tiny-mixed-defense student's 1000-question evaluation. All three actual single-GPU UUID/empty-process checks passed. This adds only evaluation, with no student or teacher training. Main400 result was 45.5%; do not extrapolate it to 1000 until all slices are complete and matched. No new reservations or extensions.

Current owners: 28527.8/srun635268 slice200; 28529.7/srun635276 slice600; 28530.10/srun635284 slice800; 28528.9/srun567428 T2 scorefit student120→old/new200. Deadlines 08:41:34 for 28527/28/29, 08:42:40 for28530. Next normal account scan >=08:45:55. Healthy existing SSH PTY45124. Do not steal the scorefit train→eval gap. Capture final results or label incomplete work at expiry.
