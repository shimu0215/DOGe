# Active three-GPU learning-loop research — Sep10 21:56 ET

## Authorization and resources
User authorizes continued research using three EXISTING A10080GB allocations, each4CPU/32GBhostRAM. No new reservation/extension, no canceling the third allocation as redundant. Stop each at its deadline; pause heartbeat only when all have expired and final records are collected.
-9795227 / gpu008: ends2026-09-11 02:03ET. preservation_control.
-9801342 / gpu010: ends2026-09-11 05:11:13ET. directional.
-9801341 / gpu019: ends2026-09-11 05:42:19ET. strong_rank.
These latest user authorizations supersede old one/two-card instructions.
No subagents or goaltools. Training may use/update proxy; inference only plain merged teacher, no gate/proxy/adapter/component. User wants own greedy/sampling performance intact and student OPD gain removed, ideally atSFTlevel. Never infer success just from auxiliaryloss or SFTmatching.

## Connections and code
AuthenticatedSSH PTY78373,Hopper2,cwd /scratch/wzhao20/opd-gate-audit-run-20260909. Exactremotecommands incommentary BEFOREexecuting; remote scope /scratch/wzhao20 only. Every srun needs stdinDEVNULL or </dev/null. Use existingSSH,not reconnect unnecessarily.
Localisolatedworktree /Users/shimu/Downloads/opd-gate-audit-20260909; originalDOGe-main dirtyuntouched. Branchcodex/spinda-gate-audit-20260909. Code554f2e6 pushed/pulled; localreportcommitc11ac49pushed,remotehasn'tneededpullforreport. DO NOTmodify current train.py,train_strong.py,run_pilot.py,run_strong.py or sharedhashedMiniLLM/oldOPDdrivers whileactive. Separatefile fornewexperiment.
Design LEARNING_LOOP_DESIGN.md. Chronological setup/failurelog LEARNING_LOOP_LOG.md.

## Running queues — neverduplicate
Outputroot results/learning_loop_20260910.
- directional: driverPID3266861,srunPID3266886,38/128updatesat21:56,phase train.
- preservation_control: driverPID3266862,srunPID3273001,26/128updatesat21:56,phase train.
- strong_rank: driverPID3288965,srunPID3289290,phase train;4/128updatesat21:56, realupdatesverified,noerror.
Driver commands: nohup "$PY" experiments/learning_loop_20260910/run_pilot.py --arm directional (or preservation_control), and run_strong.py forstrong_rank. All alreadylaunched; doNOTrelaunchunlessverifiedfailureandproperrepair/archive.
Queues {arm}_queue.json; logs{arm}_{phase}.log and{arm}_pipeline.log; models{arm}/model.
Compactread: python3 experiments/learning_loop_20260910/progress.py. Save snapshotslocallyresults/learning_loop_snapshot.json.
Eachfixedarm128updates->teacher64all3->actualOPD120seed10->teacher200all3->studentextra600:800->teacherextra600:800. Omitphasesonlyexplicitdeadlinebudget. No testscore-basedcheckpointselection. Driversabortwitherror, noauto-resume; preserve completedoutputs/failures. Separatephase recoveryneedediflaterfails.
ActualOPD unchanged audited experiments/internalize_20260910/run_opd.py/.sh, ports30003/30001/30005for directional/control/strong. Fullstudentoutputsresults/internalize/loop_{arm}_s10_*.

## Implemented mechanisms and caveats
Alloriginal7B,last4LoRAr16a32lr1e-5,128updates. Group4outcomeRL correctfinalboxed+termination reward, sampledrefKL.02, answerCE.5 (averageowncorrecttraceanswerregion andcorrectnumericanswerafterfreshproxyprocessprefix), own-originalteachertrajectoryKL1. FinalordinaryFP16merge.
Liveproxyoriginal0.5BSFT,FP32last4LoRAr8a16 (733184params), generatesfreshtrajectoryEVERYiteration, thenactuallySGDupdateslr.02clip1on24fullresponseconditionalreverseKLpositions. Futureiterationusestheupdatedproxy. This INNERlearner is NOTfullMiniLLMPPO/Adam/fullweightOPD. Full120stepOPDistheexternaltest.
Directional: h=normalizedgrad of correctNUMERICanswerCE conditionedon LIVEproxyprocessprefix+fixedfinalanswercue. NOcorrectteacherderivationrevealedtotheproxyQtarget. FDphi±.05hcomputesc=(pplus-pminus)/(2eps),e=entropyderivative; alignment=e-sum(c logpT). Hinge(alignment/sumabs(c)+.02) attop4of24middlepositions. Weight0first16,ramp2by64,then2. Onlymiddle(first32excluded,answerexcluded) trainingpositions; sparse trainingmaskdoesNOTguaranteesparseinferencechanges. QcanstillbenearzerooncorrectliveCoTs; inspectdistribution,don'tinterpretconditionallikelihoodasfullaccuracy. FDfirststepcheckepsvssemi-epscos>.9,reldiff<.5; restoreproxyweightsexactly. MathtestsPASS.
Control sameeverything withantiweight0. Sharedwarmup verified21:55: first16steps64teacherrollouts and16proxyrollouts EXACTLYIDENTICAL between directional/control; resultwarmup_pair_verification.json.
Strong: replaceantiobjectivebyforwardKLtowards ORIGINALteacher-onlytop32reversedlogits,at4of24middlepositionswithlargestoriginal-to-targetKL; skiporiginaltop2EOS/pad. Protectedall specialtokens/paddedhead. Target/rankNOproxylogitsorgradient,onlytrainingprefixfromproxy. Sameantiweightschedule/commonpreservation/closedloop. Reversalpreservesentropyandpartition,butdoesnotensuresemanticmisguidance/teacherperformance. Checkspass.

## Smoke/provenance
Corrected directional smoke COMPLETE2updates,bothteacher/proxyupdate,plainmerge;peak17.803GB;FDcos.999987,reldiff.006476;maxmergelogiterror.046875,mean.011083,argmaxsame(onepromptonly). Formaltrain.pyhash5f87b07f460ca8e6102fea51430bf9527f75aa7587e111b305e605e69c2e3fc5.
Strongsmoke COMPLETE2updates,bothupdates+plainmerge;peak17.789GB;partitionerror1.907e-6,entropy2.235e-7,specialscoresEXACT;merge max.046875,mean.007853,argmaxsame. train_strong.pyhash99f72cbe8929aa66b1e518e8f43fc775e6876dc976ea93cade3b6775e913d939.
Allsmokesused2groups2,maxteacher256/proxy192,antiweight2immediate; cappedanswersmeansNOTefficacy. Formalmaxteacher512/proxy384.
Failedinitialsmoke defaultGradScaler65536nonfinitebeforeupdate archived *.fp16_overflow (smoke/log/queuefiles); conservativescale128fixkeepsnonfinitefatal. IntermediatePASSsmoke.teacher_trace_target archivedbecauseQoncorrectteachertracewastrivialanswercopy (~1e-5); correctedBEFOREanyformaltraining. No formalphasefailureasof21:55.

## Original RL study is COMPLETE — doNOTrestart
Oldresults/rl_process_9795227 queueall12phasesCOMPLETE21:46:49ET; signalauditCOMPLETE21:48:46ET. RL_PROCESS_REPORT.md/results/rl_process_final.json finalrecords.
Originaljointstudent50.5old/51.5extra,RLonly51/53,originalteacherclean10 49.5/51.5,SFT50.5/50. JointvsRLonlyold-.5ppCI[-6,5]. No suppression. Jointteacher91.5/89.5/86.5old,92/90/92extra,original90/90.5/88old,91.5/91.5/91extra. Jointsamplingpreservationunproven.
MergedjointKL.158932 vsoriginal.174671(~9%lower), conditionalKLlogitgradL2.074716vs.084326(~11.4%lower). Mergehasnoterasedchange; thisisNOTfullstudentparametergradientorOPDupdate. Larger/smallerauxiliarysignalnotitselfsuccess.

## Followthrough
Keepallthreequeuesprogressing,checkrealaccuracy/provenance/pairedscores. Onceprimarycomparisoncomplete,timepermitsneededpairedseedconfirmationorprocess/answermechanismablation. Historicalcleanseed10OPDgainunstable; knowncleanoldseed11=53/seed12=53.5,extra11=45/12=51.5. Don'tcherry-pickcleanbaseline. Ifcandidateworks, testotherdataset/student. No blindparamscan. Endrespectiveallocationusewhenexpired; afterlast05:42:19gatherfinal+explicitpartial,commitreportandpauseheartbeatteacher.
HeartbeatteacherACTIVEevery15minupdatedforthreecards. Notifymeaningfulresults/failure/completiononly. No current successful defense claim.

Latest21:56allthreeformaltrainersactive,finiteupdates,~17.97GBpeakGPU. strong_smokecompleteandmerged; strong_rankdriverPID3288965/srun3289290. Snapshotresults/learning_loop_snapshot.json. Code/reportc11ac49pushed; activealgorithmfilesunchanged. No newefficacyendpointyet.
