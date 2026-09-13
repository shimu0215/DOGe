# Teacher 防 OPD：截至 2026-09-12 23:54 ET

当前找到的是能显著削弱 OPD 收益的普通 teacher checkpoint，尚未在扩大评测中稳定把学生压回 SFT，或只留下 1–2 个百分点收益。

所有下表学生从同一个 short-SFT checkpoint 独立开始，固定 120 步 corrected MiniLLM OPD；不是先正常 OPD 再干扰 OPD。扩大评测只增加测试题，未改变训练集、训练步数或选择 checkpoint 的方式。Teacher 候选均从原始 7B 直接微调最后两层，不用 LoRA；推理没有外挂。教师训练仅使用离线 CoT 文本、原 teacher 分布与自身正确性信号，没有 student 参数、logits、梯度或学习收益反馈。

| 条件 | 主 400 题学生 | 完整 1000 题学生 | teacher 质量证据 |
|---|---:|---:|---|
| SFT 初始 | 43.75% | 43.5% | 不适用 |
| 原 teacher OPD | 52.5% | 54.1% | 比较基准 |
| 次优候选重排 + 频繁正确性 RL | 43.75% | 47.3% | 常规 1000 题保持；额外 raw 400 题下降 4.5pp |
| 加强重排抑制 + 更强分布保持约束（anchor36） | 43.25% | **46.7%** | 常规主 400 题通过；扩展 400 题 greedy +0.25pp、sampling −1pp，点值通过 |
| 加强重排抑制 + 更强答案监督（answer4） | 43.5% | 47.2% | 常规主 400 题通过；扩展 400 题 greedy +0.25pp、sampling +0.75pp，点值通过 |
| 频繁难题正确性 RL（hardfreq） | 45.5% | 未完成 | 常规主 400 题及额外 200 题通过；raw128 少对 1 题 |
| teacher 与离线 token 分歧位置的相对分数修改 | 47% | 49.7% | 常规主检查通过 |
| 提高正确性 RL 权重（correct2） | 47% | 49.7% | 常规主检查通过 |

Anchor36 在完整 1000 题比正常 OPD 低 7.4pp，配对 bootstrap 95% CI [-10.1,-4.7]；相对 SFT 仍高 3.2pp，CI [0.1975,6.2]。正常 OPD 的 10.6pp 收益约减少 70%，但尚未达到残余 <=2pp 的阶段目标。它比已有 frequent-RL 47.3% 低 0.6pp，CI [-3.0,1.8]，尚不能确认优于旧方案。Teacher 完整扩展质量尚未补齐，因此也不能替代旧方案成为已验证的全面最佳结果。

方法的具体操作是在离线负例的过程位置上保留原 teacher 的最大 logit，重排其下 16 个候选的分数，用后训练拟合该目标；正例上的原分布约束、最终答案监督以及 teacher 自身正确性 RL 用于维持生成能力。负例损失先暖启动，再逐渐增大。理想目标保持最大项和概率多重集合，并不保证训练后的 checkpoint 保持这些性质，更不保证学生下降。应把分布恒等式与实际效果分开表述。

最新诊断在相同 95 条离线前缀、相同 token 位置上比较新旧 teacher。Answer4 对非首选的已观察 token 提分略强，也更扰动 teacher 自身前缀；anchor36 与旧方案的这些指标接近。这只是描述性证据，不能把分歧 token 当作错误步骤，也不能证明这些分数变化导致了学生下降。Hardfreq 的 64 组自身 RL 中有 11 组非零 advantage，但其主 400 题结果没有显示更好效果，不能因为 RL 信号增多就宣称有效。

后续有新 GPU 时，优先补齐 anchor36 的完整 teacher 扩展质量与 answer4 的学生 1000 题，再决定是否继续。现有结果主要来自反复使用的同一评测集和一个学生训练种子，是探索性结果；尚无不同学生模型或不同 QA 数据集的泛化证据。优先完成方法，不机械重复无效方案的多种子。

具体可追溯文件：PROGRESS_20260911_0955.md；results/opd_update_20260911/repaired_teachers_paired400.json；snapshot_20260912_2042.json；此前 defenses_frequent_rl_paired1000.json 和 rl_teacher_paired1000.json。短 teacher64 补测的最终状态另见后续进度记录，不能代替完整 teacher400。


上一批资源已结束：20:59:50 ET 确认账户无运行或排队 job、无 GPU step，已删除 teacher 自动监控。没有申请、延长或取消 reservation；目标并未完成。

最后的 teacher64 补测因分配时限终止。Greedy 完成 64 题，候选和原 teacher 都为 57/64；sampling 仅完成前 40/64 题，两者均为 36/40。保留这些配对结果，但不把部分完成的检查算作通过，也不代替完整 teacher 扩展 400 题。原 worker 因系统终止而停留 complete=false；外部终止证据与各模式数量记录在 results/opd_update_20260911/resource_exhaustion_20260912_2100.json。下次获得 GPU 时，应补齐质量检查和缺失的扩大评测后再做结论。


23:54 更新：新分配已用于继续研究，22480 明确排除。Answer4 缺失学生400题已完成，完整1000题为472/1000=47.2%，相对SFT高3.7pp（配对95% CI [0.7,6.8]），相对正常OPD低6.9pp（CI [-9.6,-4.2]）。与anchor36的46.7%相差+0.5pp，CI [-1.9,3.0]，没有确认两者差异。两个teacher完整扩展质量仍在进行。1.5B跨学生实验另见 CROSS_STUDENT_15B_20260913.md；当前正常OPD还未评测完，不能报告迁移效果。

9月13日00:22后：两修复teacher扩展400完成并通过当前点值标准，尚不是统计非劣证明。原始1.5B直接OPD第一轮验证最佳89%与初始持平，尚无跨学生防御证据；正在第二轮baseline。原研究新三种teacher训练、强目标直接诊断和两固定teacher的raw400评测已派发。

00:56补充：anchor36和answer4的raw T=1扩展400题均为86.75%，原teacher90.25%，下降3.5pp，配对95% CI均[-6.75,-0.25]。因此常规温度保持的结论不能推广到高温。当前11张授权GPU已分配为5张泛化相关、6张原研究，22480仍保留到独立Q3全流程结束。


### Sep13 01:36–01:44 density strength result and continuation

Account scan01:36:25: same12RUNNING (11authorized+22480Q3), no pending. All remainingtime5–7h; nextordinaryscan>=01:51. SSH45124hop-amd-2healthy. Two olddirectdiagnostics wholeCOMPLETE andnumericstepsgone, immediatelyreusedafterRUNNING/budget andruntimeemptyUUID guards.

New correctedpairedmain400 jointresults, sameIDs/prompt/GT/generation:
- gap.25 ALL179/400=44.75%, originalclean210/40052.5%, SFT175/40043.75. Suppression-7.75pp CI[-12,-3.5], residual+1pp CI[-3.75,5.75]. Individualold/new200=52%/37.5%, strongsliceheterogeneity. Exploratoryadaptive400, no statisticalequivalence/teacherpreservation/standaloneclaim. This is first promising strength evidence for gap-contractionwithDENSEcoverage, not sparse version.
- gap.01 SPARSE201/40050.25%, vsnormal-2.25pp CI[-5.5,1], vsSFT+6.5pp CI[1.75,11.25]. Moreamplitudealoneinsufficient; don't investinbalancingthesparseversionassumingstrong.
Snapshot results/opd_update_20260911/snapshot_20260913_0142.json includesworkerrecords/read-onlyQ3 andjointpairedresults.

NEW28527.1 gpu015sameUUID2c684969-2b10-8c8b-905d-917fb50bcbd9 srun3548877hop-amd-2: run_static_tail_round2_gapdense.py --job28527. Port31255. Original7Blast2direct/noLoRA, existingfixed0.5BofflineCoT pool/no newstudent feedback; gap.25count16 targetramp anti6/anchor36/answer4/ownRL.5every1,64steps. Setnegativepositions32 soall uniformlysampled32processpositionscontribute, removinghardtop8mining: unbiasedper-contextestimateofdensepositionmean, NOTevaluatingallpositionsineverytrainingstep. Existingtrainerreused, newdriveronly; GPUactual2smokeplainexportPASSED, formal64running. Afterfullteacherqualitychecks, independentfresh0.5BOPD120/main400onlyifmainpointpass. No statementthatdensefitisalreadysuccessful.
NEW28529.1 gpu015UUIDe70ee1d6-21e0-11e6-ec61-5ee3b691bb77 srun3548123hop-amd-2: existingrun_oracle_tail_gap01.py --targetall --port31253. Densegap.01directstrengthtest, fresh0.5B120/main400. CPUchecks andactual2smokePASSED, formalrunning. Workersstatic_tail_round2_gapdense_28527_worker.json/oracle_tail_gap01_all_28529_worker.json; dispatch_gapdense_28527.json/dispatch_gap01all_28529.json. Original6/generalization5resourcebalanceunchanged.

Threeearlierround2teachersamplitude8/wide64/gapquarter allmain400pointpassandextra200pointpass, nowexternalstudentOPD. Raw128original93.75: amplitude8same93.75,wide64 88.28125(-5.46875),gapquarter89.0625(-4.6875). Rawis supplement,notveto; amplitude128passnotexpandedhighTproof. Don't reportstudentoutcomesbeforefinished.

Cross1.5Bround2actualfiniteupdatesreachedMiniLLM228/240,FKL206/240. No cleanpositivebaselineyet. BaseMiniLLM120saved, validationupnext; BaseFKL102/120. CoderfixedSFTvalepoch1=43,epoch2=48; selectedepoch1byexistingclosest40rule, cleanMiniLLM92/120. Allsamefamilyunseencheckpoints,notcrossfamilyproof. Numericoracle23371actual80/120healthy.
22480readonly: identity46/100 vsSFT43, dense_v2 48/100 vsidentity46; nowunprotected_internalized_opd, furtherdense_top16pending. WholeparentNOTcomplete, keepreserved. No q3filemutations. Latestuserstylecontrolsoptional/deferstrictness, creativityandstrength-firstprioritized, coreteachertrainingconstraintsunchanged.
