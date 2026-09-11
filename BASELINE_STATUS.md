# Dedicated clean OPD baseline — Sep11

Latest user request: devote one existing GPU to making clean OPD yield a meaningful stable gain. Try stronger OPD from the full SFT model first; if weak, reduce SFT training (roughly45% starting accuracy is a hypothesis, not a required fabricated result). Defense work continues on other GPUs.

Allocated9817267/gpu023,1A10080GB4CPU32G,ends2026-09-11T08:09:27-04:00. Existing KL-only defense pipeline retains GPU through its final teacher_fresh600. New baseline parentPID220058 launched, verified phasewaiting_kl_only_pipeline, no additionalGPUstep. Do NOT duplicate or allocate this card to another defense after KL finishes.

Remote /scratch/wzhao20/opd-gate-audit-run-20260909; SSH PTY78373; PY and model/cache exports unchanged. Code613c8e0pushed/pulled. Newexperiments/baseline_20260911 files pipeline.py/run_opd.py/opd.sh/evaluate.py/audit_inputs.py, syntaxchecked. SharedactualOPD and currentdefense scripts untouched. Pipeline logresults/baseline_20260911/pipeline.log; queue.json has phase/command/PID/evals/comparisons. No blanket rerun if queue fails; inspect thenrecovernewlabel.

CPU audit_inputs.py PASSED. Existing raw project-preSFT evaluation found at /scratch/wzhao20/DOGe-official/outputs/qwen2_5_0p5b_instruct_sft_7b_cot_gsm1000_correctonly_20260908/raw_gsm_test200/gsm8k-results.json. Identical200questionIDs/prompts/gold/generation to14BSFT andcleanOPD. Corrected scoresraw40.5%,SFT50.5%,cleanOPD49.5%. Thus user full rollback target40.5%, halfSFT-gain target45.5% on this slice. SFT gain10ppCI[3,17]. These targets are slice-specific and endpoint goals. Rawcheckpoint is vendorQwen2.5-0.5B-Instruct, not a never-instruction-trained base. Remoteinput_audit.json records predictionscores, trainingquestions, filehashes and modelpaths. No clarification remains pending.

Validation: GSMtrain7000:7128,128questions, runtimequestionoverlap assertion against1000OPDtrainingquestions. SFT source1000CoTids verifiedwithin0:1000,770correctonlyexamples. Confirmationtest1000:1200,200questions. Existingoldtest0:200 is explored and not used to select newbaseline configuration. Same original evaluator/codehash and corrected numeric scoring.

Sequence after prerequisite:
1. Verify actualsingleallocatedGPUfree; input/codehashes unchanged.
2. Evaluate originalstudent andfull5epochSFT onvalidation.
3. FullSFT→original7B OPD240 lr1e-6, samePPO1/rollouts8/batch2accum2/max640/explicitrawsampling/conditionalKL settings; save120/240, evaluatebothvalidation. Newrunpath andlabels, port30301. Trainingcodehashes auditedbefore/after. This changes budget+LR versusold120lr5e-7, notpureone-variableablation.
4. If bestvalidationgain<3pp, run original train_gsm_cot_sft.py read-only onexisting770trainrows with2epochs,batch4accum4,lr2e-5,seed42; saves1/2epoch. Chooseclosest45% onvalidation andreportactualscore. Twoepochcosineschedule differsfrom original5epochprefix. Same240lr1e-6OPD fromselectedshortSFT; compareinitvs120/240.
5. If bestgain>=3pp, validationselectbudget120or240 andconfirmselectedinitial/OPD ontest1000:1200. Ifremainingtime>7200sec, runseed11samechosenbudget/initial andevaluateval/test. Exploratorypositivegateisnotstabilityproof. Do not claim success from a noisy128questionpointestimate.
6. If research_sequence_complete andGPUstillallocated, inspect andcontinuebaselineopenresearch; don't let it idle. Candidate nextdirections GKDforwardKL onstudentcontexts, longerrollouts/truncationdiagnostics, promptbudget/teacherconsistency, orinitial7BCoTSFTcheckpointalreadyonHopper. Prioritizewithactualevidence.

CurrentonlyCPUpreflightandwaitingdriververified. ActualbaselineGPUtraininghasnotstartedat01:46ET. FailedoriginalcontrolOPDelsewhereisexpectedpreemptionandmustnottriggerrestart. Otherdefensequeueprogress inOPD_UPDATE_STATUS.md. HeartbeatteacherACTIVEevery5min nowexplicitlyownsbaselinealongsidefourdefensecards.

02:01ET: KL pipeline complete, baseline PID220058 automatically took gpu023. Device check/raw_val completed; full_sft_val active. Raw heldout train128 score64.0625%; this is not comparable to old test20040.5%. Current short-SFT closest45% validation heuristic cannot establish test45%. Subsequent research should select relative to within-slice raw/full gain or test both1/2epoch on validation, retaining original executed protocol. Do not silently alter active code or misstate its criterion.

03:24ET更新：fullSFT旧OPD标签120验证61.71875、240验证63.28125，初始64.0625，未提升。新2epoch SFT已完成，一/二epoch验证56.25/57.8125；旧选择规则选checkpoint49，baseline专卡已在short_lr1e6_s10。增加GPU001两路corrected MiniLLM/FKL完整SFT各240，以及新GPU011 short-FKL（同checkpoint49，384回答token）/long-FKL（完整SFT，768回答token）各240，烟测均通过。新对照使用真计数+FP16teacher+FP32logp，旧baseline仍原119/239计数+BF16teacher，不能混为单变量比较。长rollout记录有效生成token，更新数相同不等于token预算相同。

04:22ET：corrected完整SFT FKL240实验整条完成，val12057.8125/24059.375 vs64.0625；按验证选240，oldtest20045.5 vsSFT50.5（−5pp，20wrong→right/30reverse，paired95CI−12至2），未修改teacher本身导致退化，不能算防御成功。新gpu012两卡获批到12:12:15，总11GPU无pending。独立RKL/即时PG基础控制各lr5e-6/480真实更新，同完整SFT起点；无未来reward累加/无whitening/无复合loss，源自经典条件RKL及TML逐token机制，不宣称完全复刻整个trainer。解析梯度与padding检查PASS，新卡烟测待完成。另gpu001空卡接strongrank teacher与修正MiniLLM240的120/240配对，保持基线和防御参数严格相同。

04:31ET弱SFT线索：checkpoint49起点同train12856.25%，旧OPD12057.03125、24064.84375。240增益8.59375pp，19纠正/8退化，paired95CI[.78125,16.40625]（探索选择后不作为确认显著性），按既定规则在test1000:1200确认中。不能转换成oldtest45→50，也不能与raw/full同片64.0625忽略比较。GPU029之后将用强排序teacher做同弱起点/同legacy240协议配对，并补BF16teacher自身200，脚本short_rank_defense.py parent793975等当前FP16pair整条结束。

04:49ET新片确认seed10：test1000:1200初始shortSFT41%，OPD240后47%，+6pp，30纠正/18退化，paired95CI[−.5,12.5]，p=.1114。validation与新片同方向但稳定性未确认，已按预设启动short_lr1e6_s11。raw/fullSFT在此新片锚点待GPU019末段预算允许补测；不能假设短SFT41%仍优于raw。
