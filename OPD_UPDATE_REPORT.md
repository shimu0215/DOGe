# 新一轮实际更新方向实验

2026-09-11 00:16 ET：三组两步冒烟实验与权重合并全部通过，已进入64步正式训练。三组分别为PPO+条件KL的一步方向干预、强重排加当前自身轨迹保护、匹配保护对照。无效果结论。

数学检查与项目原PPO损失函数对照通过。GPU有限差分半步检查通过，峰值显存18.6GB；独立训练题的实际一步答案损失变化与Taylor预测相差约0.2–0.6%，说明局部近似在这两个样本上吻合，不代表抑制有效。

最终仍用合并后的普通teacher权重运行原协议的OPD120。代理训练只是一轨迹LoRA SGD，不是完整多轨迹Adam OPD；跨题直接答案损失也不是采样推理准确率。详细设计和提前淘汰规则见OPD_UPDATE_DESIGN.md。

00:21 ET：三组正式训练已到15/14/14步，梯度有限，显存峰值约19.7GB。干预前8步的teacher、proxy、自身保护轨迹完全一致，支持共同训练部分可比。新增获批gpu023已加入KL-only抗学习目标消融，设备检查通过，正在两步冒烟阶段。它的proxy更新仍保留PPO+KL，区别只在抗学习目标。暂无teacher或学生效果结果。

## 00:54 ET：四组teacher64完成

| 方案 | greedy | 普通sampling | raw |
|---|---:|---:|---:|
| 原teacher | 90.625% | 92.1875% | 82.8125% |
| PPO+KL方向干预 | 92.1875% | 90.625% | 84.375% |
| 保护对照 | 90.625% | 92.1875% | 82.8125% |
| 强重排+自身轨迹保护 | 92.1875% | 89.0625% | 85.9375% |
| KL-only抗学习消融 | 90.625% | 90.625% | 84.375% |

强重排普通sampling低3.125pp，配对95%区间[-7.8125,0]；两个方向方案低1.5625pp，区间[-4.6875,0]。未触发预设的粗筛淘汰门槛，但不能确认teacher无损。四组均已合并并进入实际OPD120seed10，当前进度41/35/39/17步。没有完整学生效果结果，不据辅助训练指标判成功。仍有4张已分配卡，全部在用。

## Sep11 01:30 ET interim actual OPD endpoints
On previously explored old200, seed10, unchanged120stepOPD: full_update52.0%, rank_live51.0%, kl_only48.5%; originalteacher clean49.5%, SFT50.5%. Paired candidate-minus-clean95% intervals respectively[-2,7],[-3.5,6.5],[-6,4] percentagepoints. These are completed student endpoints but not yet completed joint teacher/student evaluations. Recovered preservation control and extra200 pending. No established suppression or teacher preservation claim. KL-only is prioritized for a gentle schedule/direct-last-layer follow-up because of the point estimate and latest user preference, with selection caveat.

Teacher old200 evaluation subsequently completed for full_update:90.5/90.5/88.5% (greedy/ordinary/raw), and rank_live:90/90.5/86.5%; original90/90.5/88%. Their student52/51% endpoints do not demonstrate suppression. KL-only teacher200 and all extra200 slices still pending at 2026-09-11T01:35:50.272780-04:00.

Further completed studentextra200: full_update49%,rank_live50%,kl_only51%, againstclean51.5/SFT50%. This weakens an interpretation ofKLold20048.5% as a robust effect; no stable suppressionclaim. KLteacherold20090/90.5/87.5%vsoriginal90/90.5/88. Newly auditedrawproject-preSFTold20040.5% yields halfSFTgain45.5%; current48.5%candidatehasnotmettheclarifiedgoal. Allteacherextraandcontrolrecoverypendingat2026-09-11T01:46:11.018245-04:00.

Update02:01ET: full/rank/KL all teacher and student expansion evaluations complete. Pooled400 student50.5%,50.5%,49.75%; originalclean50.5%,SFT50.25%. None meets clarified target, no established suppression. Extra200 teachergreedy/ordinary/raw: full91.5/92.5/91,rank90.5/92/90.5,KL91.5/92/91 versusoriginal91.5/91.5/91. Recoveredcontrolold200student53%, expansionpending. Directlastlayer(no teacherLoRA)protect/gentle both2stepGPU smoke/exportPASS andformal20/64; this is implementationverification only. Newdirectrank gentle schedule queuedbehindcontrolrecovery. Dedicatedcleanbaseline active.
