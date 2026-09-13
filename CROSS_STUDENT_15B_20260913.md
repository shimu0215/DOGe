# 1.5B 未参与负例生成的迁移测试

用户要求：先建立 Qwen2.5-1.5B-Instruct 的有效 OPD baseline，再测试只用 0.5B CoT 调整的固定 teacher 是否仍能削弱学生；测试集限定 100 题。

学生直接从官方 `Qwen/Qwen2.5-1.5B-Instruct` 开始，本轮不增加 SFT。官方 revision 为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。原缓存权重链接缺失，因此下载到远端项目 `results/cross_student_20260913/qwen2.5-1.5b-instruct`，不修改旧缓存或旧模型目录。

Tokenizer 全映射和聊天模板已验证与原 0.5B、7B 一致；1.5B 与原 0.5B 的输出词表都是 151936，7B 为 152064，沿用已审计的 teacher 词表裁剪。实际两步训练还需确认词表路径、数值稳定性、FP32 master 更新与 checkpoint 保存。

固定数据：OPD 使用原流程的 1000 条 GSM8K train prompts；独立验证集为 GSM8K train[7000:7100]，测试集为 test[1200:1300]。测试仅 100 题，不扩大。初始学生和两个 teacher 条件使用相同 greedy512 BF16 评测，修正数值提取，报告对题数及配对比较。验证集用于选择，测试集不用于挑选学习率或 checkpoint；100 题结果是快速探索性证据。

两个干净 baseline 并行：原 7B teacher，学习率 1e-6 和 3e-6，各 120 步，保存40/80/120。各自先做2步 smoke 后从原始学生重新开始正式训练。种子10/PPO42/LM7、rollout8/batch2/acc2/PPO1、长度640/prompt256、T1/p1/k0 与当前 corrected MiniLLM 一致；学生 fullBF16 + FP32 master，不用LoRA。22479负责一次共享的初始验证与测试，22481可先训练，避免重复初始评测。

只根据验证集最高正确数选择设置，同分取较短步数，再取较低学习率。验证相对初始至少增加3题才接续最终干净测试；若不满足，记录 baseline 尚未建立。选定 clean checkpoint 在 test100 也必须正收益，才运行防御对照；若不提升，不把它报告为有效防御测试，也不在该测试集上重新挑选 checkpoint。

防御 teacher 固定为 `results/opd_update_20260911/static_tail_freq_anti6_anchor36_9916493/model`，它之前的 0.5B 学生1000题为46.7%。来源清单为原始0.5B-Instruct和完整CoT-SFT后的0.5B；没有1.5B负例。教师仅直接最后两层训练，普通checkpoint，无LoRA、外挂或学生参数信号。本轮不训练teacher，也不向teacher训练引入1.5B数据。防御 student 从同一个原始1.5B权重重新开始，匹配已选clean的学习率、步数及种子，不能接着clean OPD训练。

首次账户检查23:15 ET已有22479(gpu010)、22481(gpu008)两张A100，均无计算步骤；其他pending保留，不申请、延长或取消reservation。23:20:59启动两条baseline完整pipeline及独立CPU依赖协调器。协调器仅等worker完成，不进行周期账户扫描；派发时检查完整pipeline已结束、step消失、分配仍RUNNING、剩余>=2小时，实际worker再验证单卡和空UUID。普通账户监控按15分钟。

主文件：`experiments/cross_student_20260913/common.py`、`baseline.py`、`coordinate.py`、`defense.py`。远端输出为 `results/cross_student_20260913/`。每个pipeline直到所有训练和评测阶段完成才释放GPU；不能抢阶段间隙。CPU协调器的派发并不代表baseline已建立或防御已有效，必须读实际最终结果。
