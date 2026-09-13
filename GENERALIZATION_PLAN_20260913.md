# 未见学生和数学题集迁移：9 月 13 日新优先级

用户最新要求将约一半计算资源用于未见学生、随后跨数学题集的快速趋势测试，另一半推进防御机制。22480 在现有完整实验结束前保留；现已允许读取其结果用于分析，完整结束且空闲后可使用。不得抢任何完整 pipeline 的阶段间隙，不申请或延长资源。持续每 15 分钟监控直至已分配资源耗尽。

## 推断范围

未见学生上的防御成功支持迁移性，但不单独证明 teacher 在识别“自身 / 非自身 context”。要排除风格记忆，后续应同时检查自身 CoT 的风格改写、来自其他家族或更大模型的离线 CoT，并将最终 OPD 学生完全留出。多模型负例只是改善训练覆盖，不是泛化证明。训练中仍禁止学生参数、logits、梯度、模拟更新或学习收益参与 teacher 优化；只允许固定离线文本以及 teacher 自身信号。

## 当前资源和执行

1. 22479 / 22481：已启动的 1.5B 第二轮正常 OPD 完整跑完，分别更小学习率 MiniLLM 和前向 KL。不能从第一轮验证无增益推断防御有效。若仍无提升，降低此线路优先级，不继续机械搜索种子。
2. 23369：当前两个修复 teacher 的 raw400 完整评测结束后，CPU coordinator_base.py 自动接续 0.5B-Base 的完整流水线。这使 7 张当前可用卡中约 3 张用于迁移。协调器 PID70122，输出 results/generalization_20260913/coordinator_base.json。GPU worker 再验证自身 UUID、空闲和预算，整条流水线占有卡。
3. 其余四张维持已经启动的原研究完整流水线。新策略先验证不约束 teacher 正确率时的抑制强度，再投入正确率平衡。正在运行的三个 teacher 训练不重启；以后新策略按此顺序。
4. 22480 的独立 Q3 pipeline 尚在 sparse_v2 OPD，整体未完成，不占用。入口 results/q3_reorder_20260913/q3_22480_worker.json；只在整体 complete、必要各 arm 状态明确、numeric step 消失后考虑接续，不依据某个 arm 完成就接管。其测试100题上初始43、identity clean46，只有+3pp，置信区间很宽；最终需结合原400题诊断解释。

## 学生选择、格式与数据

已把 Qwen/Qwen2.5-0.5B 下载到 /scratch/wzhao20/opd-gate-audit-run-20260909/results/generalization_20260913/qwen2.5-0.5b-base，revision060db6499f32faf8b98477b0a26969ef7d8b9987。与 teacher 的完整 token→ID 映射相同（151665 tokens）；原始 EOS/PAD 是151643。该模型未参与当前防御 teacher 的离线负例生成，但与已知学生共享 Qwen 家族和预训练祖先，因此只称“未见 base checkpoint”，不能称跨家族证据。

也检查了 HuggingFaceTB/SmolLM2-360M-Instruct 的 tokenizer（只下载 tokenizer/config），revisiona10cc1512eabd3dde888204e902eca88bddb4951，49152 tokens，与 teacher 映射不同。不能通过裁剪输出词表把两者逐 token KL 当作合法对齐。若以后采用跨 tokenizer 蒸馏，必须验证 token/string 对应和目标定义，并对原始及防御 teacher 使用完全相同算法；目前不为追求跨家族标签而引入未经审计的对齐。

所有新下载显式设置 HF_HOME、HF_HUB_CACHE、HF_DATASETS_CACHE、XDG_CACHE_HOME、TORCH_HOME 到 /scratch/wzhao20，禁止使用 home 缓存。下载证据 results/generalization_20260913/download.json。

0.5B-Base 流水线 base_pipeline.py：

- 先在验证100题用原生 completion prompt 评估 raw base，并用 teacher 原生 chat prompt 评估相同问题。只作能力差距诊断，格式不同不称严格同分布对照，也不把 raw base 的 zero-shot 表现视为能力上限。
- 使用独立目录软链接原 base 权重；采用 teacher 的 ChatML tokenizer/停止配置准备 SFT。确认完整词表映射相同。原下载文件不改。复用已审计的14B正确CoT训练数据，记录哈希和 token 范围。两轮 SFT，在独立验证集选择最接近40%的 checkpoint，同分更早 epoch。这是为 OPD 保留学习空间的预设规则，不根据测试集降低初始学生。
- 相同 GSM 训练1000 prompts，选定 SFT 初始学生 → 原7B MiniLLM OPD120步，验证40/80/120。验证相对 SFT 至少+3题，再对固定test1200:1300的100题评测初始与选中clean模型。不能把 SFT 收益算成 OPD 收益。
- clean test 确认正收益后，从相同 SFT checkpoint 独立训练，匹配 LR/步数/种子，换固定 anchor36 teacher，测试同100题。Teacher 完全不更新、不加入该学生 CoT。报告 SFT / clean OPD / defense OPD 对题数和配对差值。
- 任一阶段无 headroom 或无 OPD 正收益，记录未建立 baseline，继续针对该阶段修改；不把学生不涨当作防御有效。不做多seed。

## 后续题集和优化方法

跨学生出现有效趋势后，做 GSM-only 后训练 teacher → 另一数学题集的测试，优先 MATH 中等难度子集。先核对 teacher > student，再建立正常 OPD 正收益，再测试冻结 defense。训练、验证、测试分开，小规模即可；MATH 必须用合适的数学答案等价判分，不能照搬 GSM 的末尾数字提取。难度无效时换中等难度，不假设更难就一定增大师生差距。不得从 MATH 测试题构造 OPD 训练或 teacher 修复数据。

PCGrad 原论文将冲突任务梯度投影到另一任务梯度的法平面：[Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)，[作者代码](https://github.com/tianheyu927/PCGrad)。它适合分离“保持 teacher”和“拟合干扰目标”两组梯度，不需要 student 参数。最后两层约466M参数，每份额外FP32梯度约1.86GB，显存上可能可行，需实测峰值；它不能保证有限步、Adam更新或准确率不降，更不能补救无效干扰方向。

仓库已有的单向 Adam 更新投影并非标准双向 PCGrad。旧匹配比较：主400投影44.25% vs未投影45.5%，差异不明确；投影扩展1000学生48.2%，teacher扩展greedy下降2.5pp。不能声称 PCGrad 已试过或已失败，也不应把已有投影宣传成充分解决方案。待强度诊断支持的目标出现，再用同一目标/数据/权重对照测试 PCGrad，记录冲突率、两组梯度范数和实际效果。

## 00:45–00:56 新卡和实际启动更新

又有28527/28528/28529/28530四张8小时单卡上线。没有搬迁或重复base实验：等待队列已在raw400完整结束后自动启动base_transfer_23369，srun102434。Base原生completion验证53/100、teacher nativechat同题97/100（格式差异见前述限制），目前SFT2。

当前11张授权卡按5/6划分：泛化=22479、22481两1.5B，23369 Base，28528 Coder，28530外家族前缀诊断；原研究=23372、23370、22478、23371、28527、28529。22480仍运行独立Q3，不计入这11张。最后普通账户扫描00:45:54，后续仅新派发依赖检查。

28528.0 gpu007 UUIDd1a6cb76-d866-5de8-cd07-ff11b2cc35c1，srun134666，coder_pipeline.py --job28528。新增Qwen2.5-Coder-0.5B-Instruct，revisionea3f2471cf1b1f0db85067f1ef93848e38e88c25，完整151665 token映射与teacher相同。它也是未见checkpoint、仍属Qwen家族，不是跨家族证据。原生chat验证→teacher差距→同一14B正确CoT的适度SFT→正常OPD正收益→固定anchor36对照，规则与Base相同，ports31225/31227，输出coder_transfer_28528_worker.json。

28530.0 gpu025，srun143229，foreign_prefix_diagnostic.py --job28530。已把SmolLM2-360M-Instruct完整weights下载到scratch，同前述revision。这是独立家族的功能诊断：64道验证题生成SmolLM CoT和原teacher CoT，再加teacher自身CoT空白格式归一化对照；所有文本重新用teacher tokenizer编码，比较固定原teacher和anchor36的分布变化、观测token logp变化、argmax变化，并按正确性分层。此处比较的是两个相同词表的7B teacher，不是错把SmolLM logits与teacher对齐；不训练teacher，不声称是跨家族OPD或源识别准确率。报告内容/错误/长度/风格混杂，不能凭均值差证明来源识别。当前generate_foreign。

28527.0 gpu015 UUID2c684969-2b10-8c8b-905d-917fb50bcbd9，srun134678，run_oracle_tail_gap.py --targetall --port31231。28529.0同node但UUIDe70ee1d6-21e0-11e6-ec61-5ee3b691bb77，srun134689，run_oracle_tail_gap01.py --targetsparse --port31233。两条均新策略强度诊断：分别gap压缩.25的全过程覆盖，和更激进.01的稀疏覆盖。CPU性质检查和actual2smoke均通过，fresh0.5B OPD120/main400进行中。没有teacher正确率准入；都不是部署方案或数学上限。

原三teacher64训练均已完成，正在mainteacher质量检查。此前两个固定修复teacher raw400完整完成：原361/40090.25%，anchor36和answer4均347/40086.75%，均-3.5pp，配对CI[-6.75,-.25]。不能说high-temperature性能保持，常规温度点值检查与此分开。结果/启动来源snapshot_20260913_0056.json。

01:09更新：SmolLM前缀诊断已完成，foreign平均KL变化大于self，但SmolLM仅5/64正确、teacher62/64，存在严重质量混杂，不能认定来源识别或OPD泛化。28530已接续Base的前向KL正常OPD验证，与23369共享已选1epoch SFT初始化（验证64/100），没有重复SFT，也不自动根据测试集选择。Coder初版SFT因generation config保存校验失败，已用新coder_pipeline_fixed.py重启，旧失败保留；复用其原生验证36/100和teacher97/100。新SSH45124已认证恢复，后台实验继续。

01:44更新：5张泛化卡正常运行。Coder SFT两个epoch验证43/48，按预设选epoch1(43)，clean120进行中；Base MiniLLM120已训练保存，等待验证，Base FKL仍训练。1.5B第二轮接近240步，但尚无有效基线或防御结果。另6卡原研究中，gap.25密集直接干预main400达44.75（clean52.5/SFT43.75），现在开始无学生参数信号的密集目标固化和gap.01密集强度测试。详见最新PROGRESS和snapshot_20260913_0142。

02:03更新：1.5B第二轮仍无验证增益，两卡转Base半epoch SFT和Coder FKL。Base/Coder/BaseFKL三旧120步训练成功，评测被停止符差异断言拦下；新增matched_stops评测恢复已实际通过40步配对比较，三条正在80/120验证，所有旧失败保留。当前新owner和准确paths见PROGRESS最新段。尚无跨学生防御结论。

02:25：Coder一epoch MiniLLM验证43→52，但固定test35→32，未进入防御。Base一epoch MiniLLM无增益，FKL验证64→67（40步）已按验证选择，28530新owner在做initial/clean100，正收益才匹配FKL40防御。28528转Coder半epochSFT，23369转与22479共享Base半epoch checkpoint25的FKL验证（其初始val63）。当前owners见PROGRESS，未获跨学生防御结果。

02:48：Base一epoch FKL40 fixedtest48→48无增益，停止防御。28530已转Coder半epoch学生的长rollout/no_repeat_ngram_size0基线，max1024而非640，先实际2smoke→80步验证40/80，test100正收益后才同设置防御。这是训练生成限制的排查，非已确认bug。详见PROGRESS/dispatch_0246。
