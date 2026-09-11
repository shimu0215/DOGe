# 与 forward-KL 学生对应的直接 teacher 防御

当前新增实验是研究候选，没有宣称成功。训练可用proxy；导出与推理只含teacher普通权重，无LoRA、外挂分类器或proxy调用。teacher直接最后一层233057792参数，FP16前向、FP32 master AdamW。

## 一步学习收益的局部推导

固定一批student生成的context，令student参数为θ，teacher分布为Tφ，学生损失为D=KL(Tφ || Sθ)。保持teacher不随学生更新变化，有

∇θ D = −E_Tφ[∇θ log Sθ]。

用另一组训练问题的答案CE作为query损失Q。小步SGD的θ⁺=θ−η∇θD满足

Q(θ⁺)−Q(θ) ≈ −η〈∇θQ, ∇θD〉。

把h=∇θQ/‖∇θQ‖固定，则学习收益的方向量为

A(φ)=〈h,∇θD〉=−E_Tφ[D_h log Sθ]。

通过中央差分估计v=D_h log Sθ，即[log S(θ+εh)−log S(θ−εh)]/(2ε)。teacher可以直接降低−E_Tφ[v]；A变为负值时，上述一阶近似预测query损失上升。这个推导无需PPO奖励回报或whitening导数。

实现对中间过程token选择4个敏感位置，只在这些位置传播teacher反学习梯度。选择依据是局部teacher-logit导数−T(v−E_Tv)的范数；teacher训练阶段可以用proxy方向，实际推理不存在该计算。

## 配方与保护

独立源：experiments/opd_update_20260911/fkl_update_objective.py、train_direct_fkl.py、run_direct_fkl.py、run_fkl_student.py。

反学习使用归一化A的hinge目标max(0,A/scale+0.02)。teacher lr5e-6，64步；前24步反学习权重0，随后升到0.5；归一化后的正确性policy-loss系数2、答案CE系数1、自身轨迹分布保护系数2。reference teacher仅训练时用于自身分布保护。teacher的真实greedy/普通sampling/raw先做64题粗筛，明显退化则不做student实验。

proxy为FP32最后4层LoRA的小模型，SGD一步forward KL，clip1，学习率.02；新context每轮重新生成。query使用另外两道问题的最终数字CE，未提供CoT。因此它是训练用局部代理，既不等同生成推理准确率，也不等同完整学生Adam轨迹。

实际学生评估使用已验证的corrected forward KL：完整student BF16+FP32 master Adam，teacher FP16，120次真实更新、lr1e-6、相同SFT/seed10/rollout8/batch2acc2/prompt/T1p1k0，与原teacher的forward_kl_clean240_s10中第120次checkpoint做同题200对比。不能拿旧PPO120结果当本组clean对照。只有实际student结果与teacher自身表现同时满足要求，才算候选有效。

## 验证和局限

CPU双精度玩具模型直接计算真实混合导数，与中央差分表达式对照：alignment误差5.37e-11，teacher混合梯度最大误差6.30e-14；独立query一步Taylor符号检查通过。实际GPU烟测会再比较ε与ε/2的一致性并验证普通权重导出。

上述理论只覆盖固定context、固定proxy方向、小步SGD的一阶变化。它没有证明新学生、全参数Adam、多步分布漂移或最终准确率下降；正确性保护也是软约束，必须用greedy/sampling实际评估。当前尚无此候选的GPU效果结果。

03:33ET：实际2步GPU烟测通过；方向v在epsilon与epsilon/2下cosine .9999997、相对差.00062164。独立普通teacher架构重建最大logit差0，保存权重校验通过。已经进入64步正式训练，尚无teacher准确率/实际studentOPD端点，不改变上述局限。
