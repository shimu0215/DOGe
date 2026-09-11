# OPD 流程审计（2026-09-11）

当前结论：**值得先修正、校验学习流程，再解释防御失败。已发现可复现的问题，但还没有证明它们是准确率不升或不降的主要原因。** 核心蒸馏方向没有发现写反；不能据此说整个 OPD 没有工作。

## 已确认的实现问题和不一致

|项目|证据|影响与边界|
|---|---|---|
|Teacher 精度选项未生效|`train_minillm.main` 根据 DeepSpeed 配置把 `args.dtype` 改为 BF16；`get_teacher_model` 只用这个 dtype，未使用 `teacher_model_fp16`。直接执行原加载函数、拦截模型工厂，确认 flag=True 时传入的仍是 BF16。|既有 OPD 实际 teacher 为 BF16；teacher 单独评估与防御训练为 FP16。之前将 OPD teacher 描述为 FP16 不准确。不能假设精细权重修改在转换后仍有相同信号，也不能在量化前断言信号被抹掉。|
|相同 teacher/student logits 的 sampled reward 不严格为零|直接执行生产 `Reward.reward_fn` 与 `get_log_probs`，同一组 logits：BF16 的 `log T−log S` RMS=0.021036、max=0.125；FP16 RMS=0.002367；FP32 RMS=3.40e-7。|Teacher 用低精度中心化和 logsumexp，student 用低精度 log_softmax，数值路径不同。这里是合成 CPU 输入，实际 rollout 上的噪声及与防御信号的比例待 GPU 测量。|
|Padding 参与 advantage 的均值和方差|生产 `whiten` 对整个 `[batch,response_length]` 计算统计量；只在最终 loss 才 mask。保持两条有效 80-token reward 完全不变，仅补零到384，实际函数输出有效位置最大改变10.322。|损失依赖 padding 长度，而非仅有效 token；上游旧 MiniLLM 也如此，不是本项目独有改动。不能把合成输入变化当实际准确率差。应测试仅有效位置标准化或基础无标准化逐 token 目标。|
|checkpoint 标签比完成的 optimizer 更新多1|用假的两次累积 optimizer 执行完整生产 `PPOTrainer.train`：标签40/80/120分别完成39/79/119次更新，最终238次 microbatch。|确定的小偏差，不能单独解释弱学习。不是“120步都没有更新”。真实 DeepSpeed 的参数变化检查已排队。|
|Proxy 学习目标与实际 OPD 不完全一致|`update_objective.py` 用 γ=1、单条无padding轨迹；实际 CLI 未设γ，默认0.95、batch2、固定384响应位参与标准化。此前 `check_shared.py` 人为设置γ=1。|此前数学检查只证明特殊配置下的恒等式，不能证明完整生产 OPD 的匹配。这是防御训练的 surrogate 偏差，并不等价于 OPD 核心算法本身错误。|

原始输出：[CPU 审计结果](results/opd_flow_audit_20260911/cpu_checks.json)。脚本直接抽取生产函数，通过测试替身执行；共享代码哈希保存在输出中。

## 已核对的核心路径

- 采样来自当前 student，8条 rollout、PPO epoch1；rollout store 每轮清空重采样。不是一直训练固定旧 rollout。
- 位置使用 prompt 长度−1 开始的 logits 去预测第一枚 response token，未发现整段统一错一位。实际 padding/EOS 边界仍列入 GPU 诊断。pad=eos 的设计使前一位置的 mask 包含首个生成 EOS，而排除后面的填充。
- Reward 是 teacher 选中 token 的 log-probability 加 student 的负 log-probability；PPO loss 有负号。`single_step_reg` 是 `KL(student || teacher)`，teacher 分布冻结，student 概率保留梯度。
- 当前两个学习通道都来自传入的 teacher：sampled reward 和 conditional KL。旧外挂 gate 只改一条通道会被另一条抵消，因此两者都必须检查；当前普通权重 teacher 本身进入两条通道。
- `--kd-ratio 0.5` 只用于可选离线 LM/KD 分支；当前未提供 LM 数据，该参数不会把 OPD 强度设为50%，也没有一个隐含 SFT loss 把 student 拉回去。
- 先前修复的模型默认采样参数覆盖和 prompt 内聊天分隔符 mask 问题，当前实际入口均已开启修复。生产脚本仍采用 T=1、top-p=1、top-k=0，避免截断采样与未截断奖励明显不一致。
- 终点评估从独立目录中唯一的 checkpoint 标签加载，非直接复用 SFT 路径；已记录模型路径和生成配置。还会用实际权重变化验证训练链路。

## 与基础方法的对照

**MiniLLM。** 当前 PPO 与 conditional KL 的实现同官方核心 loss；差异不主要在公式方向。论文采用学习率5e-6、mini-batch64、每轮256条 rollout、4个 inner epochs、5000步；我们的旧探索是5e-7、有效batch4、8条rollout、1个epoch、标签120（实际119次更新、476个训练样本使用）。这不是严格复现实验。数据与模型不同，不能机械照搬论文预算，但现有配置不足以支持“student已到上限”的判断。当前独立 baseline 已在测试240标签、1e-6。[论文附录B.1](https://arxiv.org/html/2306.08543v3#A2.SS1)，[官方 loss](https://github.com/microsoft/LMOps/blob/main/minillm/minillm/losses.py)。

**GKD。** 在 student 自己生成的上下文上，直接优化条件分布的 divergence，不需要 PPO。GSM8K 实验中 forward KL 表现较好，是应优先加入的简单对照。不能把 `KL(T||S)` 和 `KL(S||T)` 当成同一个目标，也不能把半混合 KL 与 generalized JSD 混同。[GKD 论文](https://arxiv.org/html/2306.13649v3)，[Hugging Face GKD 实现](https://github.com/huggingface/trl/blob/main/trl/experimental/gkd/gkd_trainer.py)。

**ImitKD。** 更早的工作通过 imitation learning 在学生访问的状态上查询 teacher，支持这种基本流程本身；不需要引入复杂奖励模型或多轮元优化才能构建有效对照。[作者代码](https://github.com/asappresearch/imitkd)，[论文](https://arxiv.org/abs/2009.07253)。

**Thinking Machines 的简单 OPD。** 作者实现直接把逐 token `log T−log S` 加入 advantage，默认未来折扣为0，不使用当前 MiniLLM 这种“折扣累计后除以剩余长度再标准化”的路径。它适合作为另一种精简参照；其 LoRA 学习率不能直接照抄到我们的全参数训练。[作者说明](https://thinkingmachines.ai/blog/on-policy-distillation/)，[作者训练实现](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py)。

检索于2026-09-11，官方代码快照URL与SHA256记录在 `results/opd_flow_audit_20260911/upstream_sources.json`。没有以第三方综述或复杂新技巧作为修复依据。

## 已排的真实 GPU 审计

`experiments/opd_flow_audit_20260911/queue.py` 等待 gpu029 上 dense_update **整个流水线**完成后接用已有9817268配额（截止09:01:31ET）。进程338766；不挤占 baseline 专卡，不更改活动代码，也不重复预约。

1. 在真实生产入口跑4个标签步，记录 teacher/student 实际 dtype、DeepSpeed FP32 master 的变化、BF16模型权重变化和实际 optimizer 次数。期望验证3次更新，而不是将标签4误记成4次。
2. 首批真实 student rollout 上比较生产打分与 FP32 归一化打分；检查长度上限384、EOS和有效 token。
3. 在同一批 logits 上做强人工错误 teacher 分布的 conditional-KL 梯度响应检查；这是信号传递诊断，不是完整学生退化实验证据。
4. 原始、full_update、kl_only、direct_protect、direct_gentle teacher 在完全相同 context 和位置下分别以 FP16/BF16加载，测量原始精度差、防御log-prob信号、权重变化的大小和方向。不能预设 BF16 只会削弱，舍入也可能放大或改变方向。

目前 CPU 检查完成，GPU 诊断尚未完成；没有宣称找到最终根因。

## 后续决策

优先做独立修正版本，明确 teacher dtype，log-prob/归一化计算用FP32，修正计数，并使有效 token 的统计与 mask 一致。保留旧结果和代码，不在运行中的共享实现上直接修补。关键修正应尽量以小型同起点、同数据、同预算比较定位，不能把一揽子改动的结果归因于其中一个。

在数值与更新链路通过后，先用基础 on-policy forward KL 或逐 token reverse-KL 参照建立正对照；按实际更新次数/有效生成token预算对齐，再验证防御。若这些也不增益，再研究学习率、预算、截断、teacher与SFT来源；不通过挑选测试片区制造提升。

特别记录：新 baseline train7000:7128验证片区原始与完整SFT都为64.0625%，不同于旧test200的40.5→50.5。不能把该验证集的绝对45%当成“测试降到45%”的等价标准；后续弱SFT选择应看同片区相对增益或比较1/2epoch两个起点。

Update02:31ET: 新双卡9800275/gpu001已获批到10:25:45ET，审计已从等待gpu029迁移到这里；旧等待338766停止并标记moved_to_gpu001，不是实验失败。原queue.py与标准库冲突造成首次新worker导入失败，已重命名audit_waiter.py并用gpu001_recovery.py独立r2标签恢复；原失败日志保留。并行另一张卡实际评估original/full/KL/direct两组teacher的BF16 greedy/普通/raw64题，与对应FP16及原始BF16配对。7卡现均分配给工作，启动校验见最新运行记录。

## 02:48ET：真实 GPU 审计完成及基础对照接续

生产入口四标签确实完成3次 DeepSpeed optimizer step：第一次 warmup 学习率0，随后 FP32 master 与被采样 BF16模型权重都有非零变化。因此排除“完全没训练”，但旧120标签实际119次调用、第一次零学习率的预算口径须保留。实际 teacher/student 均BF16，FP16标志无效已在真实对象上确认。

首批8条 rollout 有2条达到384 token上限，长度194–384；样本太小，不能推断全训练截断率。前两条743个有效token中，teacher旧打分相对FP32对齐打分RMS .02533，student .000737，缩放后reward差RMS .05062。teacher额外词表质量在64个位置约2e-8，不能支持词表不对齐是主要原因。人工错误teacher分布确实显著改变conditional-KL logit梯度；这是信号链路检查，不是学生退化结果。

同四条context/64位置精度探针：full_update的FP16修改logp RMS .00741，BF16 .03148，方向cos .424；KL-only .00884/.02974，cos .498；direct_protect .04660/.03125，cos .785；direct_gentle .02983/.02709，cos .683。BF16改变方向，也可能放大信号，不能说它单纯抹掉防御。模型自身原始FP16/BF16 logp差RMS .04119。所有数字仅对应这批context与采样位置。

独立新代码 `experiments/opd_corrected_20260911`（d8a4c7d）包含两路：修正MiniLLM（显式FP16teacher、FP32概率、有效token whitening、真实更新计数）；基本on-policy forward KL，同teacher精度与计数，没有PPO loss。学生仍BF16+FP32 master Adam，T1/p1/k0、相同prompt/rollout/batch、lr1e-6。两路各先4次真实更新验证，再240次、120/240在同train验证集选，随后旧test200探索，有>=3pp验证增益才进新test片区。与旧baseline属于一揽子修正比较，不能归因于某个单项。

CPU数值检查通过：FKL解析梯度最大误差7.45e-9，自teacher梯度约3.50e-9，额外31padding不改变有效advantage，logp输出FP32。GPU烟测待完成。GPU001双卡独占步骤排队，parents440875/440876，records `results/opd_corrected_20260911/{minillm,forward_kl}_{queue,worker}.json`。它们接续已有诊断，没有修改任何活动/共享旧训练代码。

## 03:07ET：FP16旧reward的padding溢出已定位并恢复

仅切换teacher FP16的初次pair在首批reward断言失败，尚未optimizer更新；原始失败日志保留。真实同两条cached rollout诊断：所有原始logits有限（约−34.4至44.9），mean/中心化也有限；只有25个padding位置的logsumexp乘mask产生NaN。原因是旧实现先把padding logits置0，FP16下对152064个exp(0)求和溢出，再用0乘inf。CPU复现了相同问题，BF16则不溢出。它不是普通有效上下文上的teacher崩溃，也不证明旧BF16训练因此失败。

独立恢复版fp16_mask_entry.py只将next_state_value乘mask替换为torch.where选择0，保留所有有效位置的旧FP16计算、旧padding whitening和119更新计数。CPU验证有效行逐值相同；新GPU四标签完成3次真实更新，已进入原teacher120标签正式训练。该配对准确名称是“FP16 teacher＋NaN安全padding屏蔽”，不是完全单变量；先原teacher后direct_protect，同起点/seed10/预算/评估片区。脚本fp16_mask_pair.py parent500024/worker2706962，gpu029完整持有，records fp16_mask_pair_r2_*.json，新的fp16_mask_r2_*标签。旧fp16_pair.py parent460379/worker2705998失败已处理，不可直接重启。

原始诊断：[fp16_numerics](results/opd_corrected_20260911/fp16_numerics.json)。全部BF16teacher64评估已完成并保存：[BF16评估](results/opd_corrected_20260911/bf16_teacher64_complete.json)。这些仍是小样本诊断，不是防御联合成功。

03:46ET首个精度配对端点：FP16teacher＋安全padding的原teacher OPD后student old200=49.5%，与legacyBF16clean=49.5相同点估计，但各13题相反变化、paired95CI−5至5pp；对SFT50.5为−1pp。小样本单种子尚未显示baseline改善，不能称精度完全无影响。对应防御teacher仍待同协议运行。
