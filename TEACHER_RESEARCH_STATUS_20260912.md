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


### Sep13 01:56–02:03 whole results, paired-stop repair and all freed cards reused

Lastaccountscan01:56:24 all12RUNNING/no pending; next>=02:11. SSH45124hop-amd-2healthy. Mainround2wholecomplete: wide64 177/40044.25% vsSFT43.75/clean52.5; delta clean-8.25 CI[-12.75,-3.5], residual+.5 CI[-4.5,5.5]. Amplitude8 187/40046.75%, clean-5.75 CI[-9.5,-2], residual+3 CI[-1.75,7.75]. Gapquarter192/40048%, clean-4.5 CI[-9,0], residual+4.25 CI[-.75,9.00625]. Allthreecurrentstandardpointchecks pass; wide raw128-5.47pp stillfails. Exploratoryadaptive400, notnoninferiority orpristineholdout. Numericdirectoracle183/40045.75%, clean-6.75 CI[-11.25,-2.5], residual+2 CI[-2.75,6.75]. Numericnotfittedteacher. SameIDs/prompts/GT/generation jointlyverified, rawpredictionnotlegacyaccuracy; snapshot_20260913_0203.json.

Cross three pipelines base_transfer_23369, coder_transfer_fixed_28528, base_forwardkl_28530 finishedclean120 thenfailedcomparisonatval40. EXACTcause: all100 IDs/prompts/GT identical, generation stop IDs beforeSFTcheckpoints[151643,151645], afterOPDexport[151645] (PADmetadatareplacedbyEOS); assertioncorrectlycaughtunequalinferenceconditions. No evidenceoftrainingfailure. Preserveoldfailedworkers/40evals, neverremoveassertionorpretendstopsetsidentical. New evaluate_matched_stops.py wrapsverifiedunchangedevaluator butforcesstopunion[151643,151645] explicitly; generationrecordmustmatch. New resume_matched_stops.py reusesverifiedcomplete120/masterupdates/saved40,80,120, reevaluatesallcleanval withpreOPDstopset, oldSFTvalalreadyexactset. No retraining. Samevalidation+3admission, fixedtest100positivebeforeindependentfixedanchor36studentOPD, no testselection. Existingtraining stop behaviorunchanged; thisrepaironlystandardizes evaluation.
-23369.2 base_transfer_23369_stopfix_worker.json srun3643429 port31261 sameUUIDf4d4...; baseval40now64→61, subsequent80/120pending.
-28528.2 coder_transfer_fixed_28528_stopfix_worker.json srun3643439 port31263 sameUUIDd1a6...; Coder43→46at40, subsequent80/120pending. No cleanfinaltestyet.
-28530.2 base_forwardkl_28530_stopfix_worker.json srun3643449 port31265 sameUUIDdaf5...; BaseFKL64→67at40. ThisworkerremainsVALIDATIONONLY, comparefullMiniLLM/FKL validationbeforematcheddefenseextension. Noautomaticclean100 ordefensehere.
Allthreerepairedactual40evaluationANDcomparisoncompleted, now80, noerror; fixvalidatedbeyondAST. Newmatched_worker.py suppliesexplicitstopsforfuturepipelines.

1.5Bsecondround wholecomplete/coordinator_round2fixed.json completebaseline_needs_improvement. MiniLLMLR3e-7 val80/160/240=86/88/89 vsinitial89. FKLbest86(step80), no+3. No selectedcleanfinaltest ordefenseperformed. BothGPUssafelyreleasedonlyaftercoordinatorcomplete, no thirdround1.5B/multiseed. Current5generalizationcardsallactive:
-22479.3 sameUUIDa8c9... srun3651741 base_half_pipeline.py --job22479. Freshraw0.5BBase, matchedstopeval, same14BcorrectCoTSFT0.5epoch (legacyphasekey sft2, actualCLI/manifestepochs=.5), thencleanMiniLLM120 validations/conditionaltest+fixeddefense. Ports31271/31273. AimlowerSFTstartingpoint/headroom, notassumed40%. raw_native_valcurrently. Newdriverdistinctdirs/noexistingtrainingoverwrite.
-22481.3 UUID5f5c... srun3651750 coder_forwardkl.py --job22481, samepreviousselectedCoderSFTepoch1(43), freshFKL120/save40/80/120valonly, port31275. Explicitmatchedstops. clean120running; no newteacherdata.
Otherthreegeneralizationcardsarerepairsabove. AllSTILLQwenfamilyunseencheckpoints, no crossfamilyproof. NoMATHyet.

Originalresearch6:
-23370.2 sameUUIDb3ce... srun3645335 eval_repair_student_extension.py --labelwide64 --teacher-source/staticstudent-source static_tail_round2_wide64_23370, workerstatic_repair_extension_wide64_23370_worker.json. Existingfixedstudentextra600 toaggregate1000; notnewtraining. Currentlytest200.
-23372.2 UUID37a8... srun3645346 sameextensionlabelamplitude8/source static_tail_round2_amplitude8_23372, workerstatic_repair_extension_amplitude8_23372_worker.json. Extra600→1000, currenttest200.
-22478.2 UUID5597... srun3651759 NEW eval_round2_teacher400.py. Wholeowner testsamplitude8 thenwide64 eachraw400 ANDgreedy/sampling400, allsametest600:1000originalreferences; currentlyamplitude8_raw400. DoNOTreusebetweenmodels/modes. workerrround2_teacher400_22478_worker.json (actualspellinground2_teacher400_22478).
-28527.1 densegapfit64complete, teacher_new200qualityongoing; conditionalfreshstudent120next.
-28529.1 gap.01ALLdirectOPD120ongoing.
-23371.2 numericwholecomplete/stepsgonebeforeNEWentropy4all; srun3659089 port31277 run_oracle_entropy4.py --job23371 --targetall. OriginalteachereligibleALLvocabulary marginscontractbeta.25 towardeligiblemax, no rank reversal, specials/paddingrawlogitsprotected. Conditionaleligibleprobabilitiesequivalenttemperature4; fullvocabnotexactT4becauseprotecteditems. No fitting/correctnessscreen, knownsourceoracle. Thisseparatesrankreversalfromentropyflattening; maystronglydamageunconditionalteacher, deploymentnotclaimed. CPUtargetpropertiesPASSED,actual2smokeongoing; inspectbeforecallingformalstarted. Fournewisolatedtarget/entry/opd/driver files. SameUUIDf9eb..., no22480interference.

Allsevenfreedauthorizedcardsreusedafteroldwholefinish/failureparentexit/numericstepsgone/budget/RUNNING andruntimecuda1emptyUUIDchecks. 22480stillreservedQ3wholeunfinished, no mutations. Dispatchrecords generalization/dispatch_stopfix.json,dispatch_0205.json andopd_update/dispatch_round2_extensions.json,dispatch_entropy4_23371.json. Usernotifiedwide44.25, stopsetfailureandactualrepair,1.5Bnogainandnewbase/Coderresearch. Nofalsecrossstudentdefenseclaim.


### Sep13 02:21–02:25 extended results weaker; next baseline and teacher experiments

Account scan02:21:52 same12RUNNING/no pending, next>=02:36. SSH45124hop-amd-2healthy. Snapshot_20260913_0225 capturescomplete extensions/crosscheckandnewworkerstartup. Priorcommit e802532pushedsuccessfully.

Student1000 COMPLETE: wide64 482/1000=48.2% vsnormal54.1/SFT43.5; normal-5.9pp CI[-8.8,-3], residual+4.7 CI[1.5,7.8]. Amplitude8 501/1000=50.1%; normal-4 CI[-6.6,-1.4], residual+6.6 CI[3.5,9.7]. Bothworseexpandedthanmain400suggested; originalanchor36 46.7stillbestexpandedeligibleteacher. DoNOTcarrymain40044.25to1000. Teacheramplitude8raw400357/40089.25 vsoriginal361/40090.25 =>-1pp CI[-3.25,1.25], pointtoleranceedge, notstatnoninferiority. Worker22478wholecontinuesamplitudestandard400thenwide64raw+standard; don'treusephasegap.

Baseoneepochfullcorrected-stopvalidation: MiniLLM40/80/120=61/61/64 vsSFT64, no+3. FKL=67/58/57, val-selected40=67(+3), bestofcompletedMiniLLM/FKL. CoderoneepochMiniLLMval46/52/51 vs43; selected80(+9). FIXEDtest100initial35/clean32(-3CI[-13,7]); baselineNOTpositive, stopwithoutdefense/test-drivencheckpointreselection. Thisdoesnotprovegeneralizationfailureofthedefense. Alloldfailedstopcomparisonsretained; successfulrerunactualgeneration[151643,151645]matches. Basehalfepochnewvalidation63 (checkpoint25actualepoch.5181347), notmarkedlylowerthanoneepoch64; itscleanMiniLLMstillongoing. CoderFKLstillongoing.

Fivecompletedownersfreedandimmediatelyreusedafterwholecomplete,stepsgone,RUNNINGtime>12600 andsingleemptyUUIDchecks; same5generalization6researchbalance:
-28530.3 gpu025 UUIDdaf511d8-e962-9aed-37e9-4ccce2230390, srun3746579, base_fkl_transfer.py --job28530, workerbase_fkl_transfer_28530_worker.json. FixselectedBaseFKL40beforetestbyvalidation67>MiniLLM64. SameSFTinitial test100+clean40 test100, onlypositivegainthenmatchedfixedanchor36FKL40freshSFTstudent, port31281. No retrainingclean or useoftestselection. Currentinitial_test100.
-28528.3 gpu007 UUIDd1a6cb76-d866-5de8-cd07-ff11b2cc35c1, srun3746591, coder_half_pipeline.py --job28528, workercoder_half_28528_worker.json. FreshrawCoder withsame14BcorrectCoTSFT.5epoch, matchedstopworker, nativeheadroomresultsreusedvalidolddata, thenMiniLLM120/selectval/conditionaltest+defense. Ports31283/31285. No claimreducedSFTgivespositivebaseline, allnewdirs.
-23369.3 gpu015 UUIDf4d4e65b-a3f0-c46c-2124-12cef395c78c, srun3746600, base_half_forwardkl.py --job23369, workerbase_half_fkl_23369_worker.json. SamealreadyselectedBasehalfSFTcheckpoint25from22479validation63, freshFKL120/40,80,120VALIDATIONONLY, port31287. ComparewithsamehalfMiniLLMwhenbothcomplete; noautomaticdefense/test. Currentclean120. Existing22479halfMiniLLM and22481CoderFKLnotinterrupted.
-23370.3 gpu002 UUIDb3ce6d89-5d78-cd18-3856-7da5231708ab, srun3746609, run_static_tail_round2_gapdense12.py --job23370, workerstatic_tail_round2_gapdense12_23370_worker.json. Newdensegap.25target32uniformpositions/ramp,anti12 vsongoingdenseanti6, anchor36/answer4/ownRL.5eachstep,last2direct64/noLoRA. Actual2smoke→plainexport→formal64→teacherquality→conditionalfreshstudent120/main400, port31289. Thisstrengthensfitofalready-effectivegap.25denseobjective; notstudentfeedbackinoptimizer.
-23372.3 gpu010 UUID37a8069b-922f-4de2-143c-541de7c81a73, srun3746623, run_static_tail_round2_wideramp.py --job23372, workerstatic_tail_round2_wideramp_23372_worker.json. Combine64secondarycandidatepurepermutationwithtargetamplituderamp, anti8/anchor36/answer4/RL.5, otherwiseexisting64last2protocol. Oldwide64noamplituderamp/anchor24, amplitude8oldcount16. Port31291. Sameactualsmoke/formal/quality/conditionalstudentpipeline, no model-componentaddition.
Dispatchproof results/generalization_20260913/dispatch_0225.json. AllfiveAST/new-onlydeploy andruntimeUUID checksPASS. Twofitssmokeongoingatstartup; verifycompletedbeforeclaimingformalstarted. DoNOTrestartthesejobsnextheartbeat.

Otheractive: 28527denseanti6teacher64/standardpointpass alreadyexternalstudentOPD120; 28529gap.01ALLformal120complete, evalold20024/20012%, nextnew200ongoing (largepartialdrop, awaitfull400 beforeconclusion; earlieststrongamplitudexdensitysynergy); 23371entropy4direct120ongoing; 22478teacher400wholeongoing. 22480Q3stillwholecompletefalse phasedense_top16_opd; nointerruption/reuse. Newdramaticpartialtargetmayjustifyfuturefitonlyafterwholeconfirmation; don'tpreemptactiveworkers. Usernotifiedextendedweakening/Coderbaselinetestfailure; baseline/defensegeneralitystillunproven.


### Sep13 02:42–02:48 strong target full audit and first internalization

Accountscan02:42:24: same12RUNNING/no pending, next>=02:57. SSH45124hop-amd-2healthy; 58d51a7pushed. BaseoneepochFKL40 fixedtest10048→48, CI[-8,8], so no matcheddefense/no testreselection; workerbase_fkl_transfer_28530wholecomplete. Generalizationstillnotestablished. Basehalf/Coderhalf/CoderFKL continue; Coderhalfselectedcheckpoint25val43, samevalscoreasoneepochbutdifferentweights.

Strongdirectgap.01ALLwholeCOMPLETE: old20024/20012%, new20028/20014%, total52/40013%; clean210/40052.5%, SFT175/40043.75. Jointclean-39.5pp CI[-45,-34]; SFT-30.75 CI[-36.25,-25.5]. All400IDs/prompts/GT/generationmatch. 120actualoptimizerupdatesfinite/nonzero, finalmasterdelta1.8167282e-5; reward+regularizer each240calls/480rows/135987modifiedpositions. Predictions0empty,400uniquetexts,208boxed/192last_number,median1328chars. ThisisactualmodelscoredegradationnotNaNorallemptyoutput; mayinvolveanswer-format/lengthchanges, notprovedreasoning-onlydrop. No qualityconstraint/fitting inthisoracle. Finiteprecisionargmaxtiespossibleevenifprecastpropertychecked; don't claim actualteacherpreservation. snapshot_20260913_0246.json hasfullaudit. NewDENSE_MARGIN_INTERPRETATION.md givesconditionalratio/entropyargument, uniformsamplingvsmining distinction, explicitlynooptimality/benchmarktheorem.

Afterwholecomplete/stepsgone/RUNNING>10800/cuda1emptyUUID:
-28529.2 gpu015 UUIDe70ee1d6-21e0-11e6-ec61-5ee3b691bb77, srun3830306 run_static_tail_round2_gap01dense.py --job28529, workerstatic_tail_round2_gap01dense_28529_worker.json, port31301. Firstfitofstrongbeta.01dense target: anti2/anchor36/answer4/ownRL.5every1,targetamplituderamp,32uniformpositions,64steps,orig7Blast2direct/noLoRA. Actual2smoke/plainexportPASS,formal64running. Sameoffline0.5Btexts,no studentparameters. Fullteacherstandardmain400/raw128/extra200→conditionalfresh0.5BOPD120/400.
-28530.4 gpu025 UUIDdaf511d8-e962-9aed-37e9-4ccce2230390, srun3830330 coder_long_pipeline.py --job28530, workercoder_long_28530_worker.json. ReuseCoderhalfselectedSFTcheckpoint25(initialval43). Newisolatedopd_long_unrestricted.sh max_length1024/prompt256, no_repeat_ngram_size0 vsoldmax640/ngram6; verifiedargumentdefaultinAKDA2arguments.py. SamecorrectedMiniLLM LR1e-6,80steps/save40/80 andinitial2smoke, ports31303/31305. Reasonexplore384tokenrolloutcap+ngramconstraint, notassertcausality. Matchedstopsgreedy512evalunchanged. Fullval+3→fixedtest100positive→matchedlong/unrestrictedfixedanchor36independentdefense. Currentlyactualsmoke2ongoing; ensurefinite/masterupdates/OOMbeforeformalclaim. No teachertraining. Otherfivegeneralizationresourcebalanceunchanged, no retrainingpriorclean. Dispatch_0246.json.

Original3round2/currentdense12/wideramp64formalcompleted, newqualityeval; earlier28527denseanti6externalstudentOPDongoing. 23371entropy4all formalcomplete→student_test0, nooutcomeclaimyet. 22478teacher400workerwide_standard400current: amplitudeexpandedg364/40091% equalsorig,s361/40090.25vs89.5 (+.75); raw357/40089.25vs90.25(-1). Wide raw345/40086.25vs90.25(-4CI[-7.25,-1]); standardpending. Whole22478stilloccupied. Q3whole22480stillownsitscard; no touchedfiles. All11authorizedassigned.


### Sep13 03:03–03:13 Q3 completed, entropy target strong, 12 active assignments

Accountscan03:03:54 same12RUNNING/no pending, next>=03:18. SSH45124hop-amd-2healthy. 9f68ab5pushedsuccessfully. Snapshot_20260913_0311 capturesfullcorrectedpairedresults/newstartup. AllnewsourceAST/new-onlydeployment; nooldactivescriptmodified.

Directentropy4ALLwholecomplete:10/4002.5% (3old+7new), vsnormal52.5 -50pp CI[-55.25,-45], vsSFT43.75 -41.25 CI[-46.5,-36.25]; actual120finiteupdates/masterdelta2.955765e-5,0empty/400uniquetexts, sameIDs/prompts/GT/generation. Entropyflattening WITHOUTrankreversal isstrong byitself. It isstillknownsourceoracle, notstandalone/teacherqualityclaim. Thismotivatesfirstentropy4targetteacherfit.

Densegap.25anti6teacherwholecomplete:178/40044.5%, normal-8 CI[-12.5,-3.5], residual+.75 CI[-4.5,6]. MainteacherpointpassBUTextra200greedy91.5vs93.5(-2pp)failedsupplement; raw12887.5vs93.75(-6.25). Explicitlycannotclaimteacherperformancegloballypreserved. Expandedstudent1000andteacher400follows, noignorefailures.

22478teacher400previouswholeCOMPLETE:amplitudeg91/s90.25/raw89.25, originalg91/s89.5/raw90.25. Wideg91.5/s90.25/raw86.25. Standardpointpassboth, rawwide-4pp; amplitude-1pppointedge. Student1000amplitude50.1/wide48.2remainworseanchor46.7.

BasehalfMiniLLMwholecompletebest62vsinitial63,no+3; CoderoneepochFKLwholecompletebest41vs43,no+3. No test/defenseclaimed. Theircardsreleasedandchangedbelow. OtherBasehalfFKL/CoderhalfMiniLLM/currentCoderlongremaininflight.

Q3_22480 WHOLEcomplete/allarmscomplete/numericstepsgone at03:03. Readonlysummarytest100 SFT43,identity46,sparsev243,densev248,unprotectedinternalized48,dense_top16(incoriginalmax)6. Identitybaseline+3wideCI; secondaryreversaldoesnotstronglysuppressonthissliceevenwithoutteacherpreservation, includingtopmaxstronglysuppresses. No teachingqualityclaim. Other-taskq3filesunaltered/untracked. Userauthorizationpermitreuseonlyafterwhole; nowFIRSTreuse22480forsmollmcalibration, no longerreserveitindefinitely.

Sixcompleteownersreleasedandimmediatelyreusedafterstepsgone/RUNNING/remainingtime/runtimecuda1emptyUUID. Resourcesnow6generalization/6originalresearch, all12assigned:
GeneralizationNEW:
-22479.4 gpu010 UUIDa8c99944-94c9-11cd-0525-96f3423af497 srun3917327 base_long_pipeline.py --job22479, workerbase_long_22479_worker.json, ports31311/31313. ReuseBasehalfSFTcheckpoint25val63, same longmax1024/no_repeat_ngram0 asCoderlong, actual2smokePASS, formal80active. Evaluate40/80 withmatchedstopsgreedy512, val+3 thenfixedtest100positivebeforematchedindependentanchor36longdefense. No extraSFT/reusecleanstudent.
-22481.4 gpu008 UUID5f5c0036-6942-24bd-ed30-e097a1c4bdee srun3917336 coder_raw_long_pipeline.py --job22481, workercoder_raw_long_22481_worker.json, ports31315/31317. OfficialrawCoder0.5Instruct(no projectSFT), freshmatchednativeval thenlongunrestrictedMiniLLM2smoke/80/40,80val/conditionaltest+defense. Currentinitial_val. Retainsvendorinstructiontuning; no claimpretrainingbase.
-22480.1 gpu023 UUIDc7054e53-2ace-e243-47c6-5c43e7f3cb9c srun3922509 smollm_sft_prepare.py --job22480, workersmollm_sft_22480_worker.json. IndependentfamilySmolLM2-360M-Instruct, rawnativeval100, matchquestionID/GTteacherval97, thenretokenizesame14BcorrectCoTSFTtrainingtexts withSmolnativechattemplate/EOS. EachsourceID<1000/exactGSMquestion/CoTcorrectnessassertions, target labels exclude nativeprompt; prepareindependentneutralgenerationconfig/rawweightlinks (nohomedownload). FullstudentSFT1epochLR3e-5 thennativeval100. ThisisPREPARATIONONLY, no cross-tokenizerKL/no OPD/no defenseclaim; mayenablenextnativefamilyOPDbaselinewithprincipledalignment. Currentraw_val100. Q3wholecompletionverifiedbeforelaunch, noother-taskmutation. dispatch_smollm_22480.json.
Existinggeneralization23369.3BasehalfFKLval,28528.3CoderhalfMiniLLMval,28530.4CoderhalfLong80continue. Thussixgeneralizationcards.

OriginalresearchNEW:
-23371.3 gpu023 UUIDf9eb0558-9328-c940-8173-8dcc177f880d srun3917345 run_static_entropy4_dense.py --job23371, workerstatic_entropy4_dense_23371_worker.json, port31319. NEWtrain_static_entropy4.py original7Blast2direct64 usingoracle_entropy4_target.TailPermutation target(allvocabularygap.25,no reversal), anti1anchor36answer4ownRL.5every1,targetramp/uniform32negativepositions. No studentmodel orparameterfeedback. Actual2smoke/plainexportPASS, formal64running, qualitychecks/conditionalfreshstudent120/400. Metadata targetsourcehashupdated; no entropy-preservationclaim.
-22478.3 gpu011 UUID5597ad5f-bf36-02b5-b332-c18284fdf6d0 srun3917364 eval_gapdense_teacher400.py --job22478, workergapdense_teacher400_22478_worker.json. Fixed28527denseanti6teacher raw400andstandardg/s400 same600:1000references; wholeownercurrentgapdense_raw400. Neededgivenexistingextraqualityfailure, no newtrain.
-28527.2 gpu015 UUID2c684969-2b10-8c8b-905d-917fb50bcbd9 srun3917373 eval_repair_student_extension.py --labelgapdense --teacher-source/staticstudent-source static_tail_round2_gapdense_28527. workerstatic_repair_extension_gapdense_28527_worker.json, extra600to1000fixedstudent no newtraining. Currentstudent_test200.
Existing23370.3dense12,23372.3widerampteacherquality→conditionalstudent;28529.2gap.01densefirstfitteacherqualitycontinue. Sixoriginalresearchcards.
Dispatch_0308.jsonsavesfivecommands;smollmseparateproof. Neverrelauncholdcompletedextensions/baselines/Q3. Laststartupnoerrors, Smol/rawCoderstillinitialevaluation, completeformalclaims onlyBaseLongandentropyfitwithactualsmokepass. Usernotifiedentropy2.5/dense44.5withqualitylimits/Q3releaseandcrossfamilySFTpreparationdistinction.


### September 13, 03:30–03:44 ET: expanded dense result, PCGrad and cross-family calibration

Dense beta=.25 / anti6 expanded student evaluation is now complete: 494/1000 (49.4%), compared with SFT 43.5% and normal OPD 54.1%. Its suppression is 4.7 pp (paired 95% CI −7.6 to −1.8), but residual OPD gain is 5.9 pp (2.8 to 9.1). The promising 44.5% main400 result did not persist at this larger evaluation. Teacher supplementary400: greedy 90.5% vs 91%, standard sampling 89% vs 89.5%, raw T1 84.75% vs 90.25% (−5.5 pp, CI −8.5 to −2.5). It is not an improved preservation/suppression solution.

The first stronger beta=.01 dense internalization failed the main teacher criterion: old200 sampling89% vs90.5% (−1.5 pp); new200 sampling85.5% vs86.5% (−1). Raw12881.25% vs93.75%. Therefore no external student training was launched for that checkpoint. Strong direct-output oracles remain evidence of target strength, not successful internalization.

New original-last-two-layer trials: 28529.3 static_gap01_pcgrad (srun4028327, port31333) and 28527.3 static_gap01_anchor72 (4028345, port31335) both passed actual two-step FP16 training and ordinary-model export/reconstruction checks and entered formal64. PCGrad splits weighted preservation (teacher-own RL, live/static anchors, answer CE) and interference gradients, symmetrically projects both when their dot product is negative, sums, then clips and feeds FP32-master AdamW. CPU explicit-formula tests passed. This is genuinely two-objective PCGrad; the earlier one-sided Adam displacement projection is a different method. PCGrad before Adam does not guarantee the realized update or finite-step accuracy. The matched preservation-weight alternative doubles anchor36→72 and own-RL .5→1, keeping anti2. Neither uses student parameter-derived signals.

22478 supplementary teacher400 worker finished; numeric steps disappeared. Its card was reassigned to static_entropy4_pcgrad_22478 (22478.4, srun4062121, port31337): same all-eligible .25 gap contraction without rank reversal as the entropy4 trial, anti1/anchor36/answer4/ownRL.5, uniform32 process positions and gradual target. CPU PCGrad checks passed; actual two-step smoke was running at03:44. Full teacher quality and conditional independent student evaluation follow. No success claim yet.

Cross baselines: Base half-SFT forward-KL best63/100 equals starting63; Coder half-SFT short MiniLLM best44 vs43; Coder long MiniLLM LR1e-6 best45 vs43. None meets the prespecified validation+3 gate; no fixed test or defense comparison was run. Released cards now run Base tenth-epoch SFT→short MiniLLM120 (23369.4, srun4023632, ports31321/31323), Coder half-SFT long forward-KL80 (28528.4,4023641,31329/31331), and Coder half-SFT long MiniLLM80 LR3e-7 (28530.5,4023650,31325/31327). Both new Coder actual2 smoke checks passed; formal runs active. Base tenth completed native calibration/SFT checkpoint5 and entered clean120. Existing Base-long22479.4 and raw-Coder-long22481.4 continue. These are attempts to establish positive baselines, not evidence of transfer.

SmolLM2-360M-Instruct native raw validation is7/100 versus originalteacher97 on the same IDs/answers. Preparation caught two incorrect legacy14B CoTs, IDs409/835, before any SFT. New filtered preparation keeps768/770 and records exact rejected answers; historical corpus and past results remain untouched. Native chat/EOS retokenization, question/ID and target-label checks retained. One epoch full student SFT completed:6/100 (−1 pp vsraw, CI −6 to4). This is not a useful OPD baseline yet. After whole worker completion/numeric steps gone,22480.3 (srun4056445) now tries three epochs from the original raw checkpoint with the same768texts, then validation100. No naive cross-tokenizer logit cropping; cross-family OPD remains unimplemented/unvalidated. Q3 completed before the first reuse, and its files remain untouched.

Latest account scan03:42:47:12RUNNING/no pending. Next ordinary scan no earlier than03:58. Six cards assigned generalization, six core research. Other ongoing core owners:23370.3 dense12 external student evaluations;23372.3 wideramp external student evaluations;23371.3 entropy4 first-fit teacher supplementary evaluation. Retain whole-pipeline ownership and check completed manifests, steps, remaining time, and empty actual UUID before every reuse. Snapshot filename0349 contains actual03:42 timestamp (filename is a label, not wallclock evidence). New dispatch records and snapshots saved. No new reservations, no extra student seeds, no MATH defense claims.
