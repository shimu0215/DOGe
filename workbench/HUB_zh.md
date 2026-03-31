# Workbench 总说明

这个目录是 Codex、Claude Code、OpenClaw，以及相关 GUI / terminal AI 会话的共享协作基地。

## 身份规则

- 如果你还没有固定名字，请先按这个格式给自己起名：
  `<ai名-丰缘地区宝可梦名>`
  例如：`codex-水跃鱼`
- 每次面向用户回复时，都先提到这个名字，方便用户确认你没有脱离这个共享上下文。
- 在 handoff 和任务记录里也使用同一个名字。
- 在这个目录开始做实际工作前，先把自己登记到 `/Users/shimu/Downloads/DOGe-main/workbench/AI_REGISTRY.md`
- 你的登记名字必须唯一，不能和已有名字重复。
- 即使你也是 Codex、Claude Code 或其他已经出现过的 AI，只要你和之前那个会话不共享记忆/上下文，就必须起一个新名字。

## 默认读取顺序

除非当前任务明确需要更多历史，否则先只读这三个文件：

1. `/Users/shimu/Downloads/DOGe-main/workbench/current/active_tasks.md`
2. `/Users/shimu/Downloads/DOGe-main/workbench/current/latest_handoff.md`
3. `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/ssh_status.md`
4. `/Users/shimu/Downloads/DOGe-main/workbench/AI_REGISTRY.md`
5. 如果你的工作涉及数据集，读取 `/Users/shimu/Downloads/DOGe-main/workbench/DATA_REGISTRY.md`

之后再按需读取具体的 knowledge、tasks 或 dashboard 文件，不要默认全量扫描历史。

## 目录结构

- `/Users/shimu/Downloads/DOGe-main/workbench/AI_REGISTRY.md`
  所有进入这个工作区的 AI 会话登记表。
- `/Users/shimu/Downloads/DOGe-main/workbench/DATA_REGISTRY.md`
  数据采集、删除、废弃、迁移及存储位置的共享登记表。
- `/Users/shimu/Downloads/DOGe-main/workbench/current/`
  当前状态层，只放短而新的信息。
- `/Users/shimu/Downloads/DOGe-main/workbench/knowledge/`
  可复用经验、稳定流程、通用操作方法。
- `/Users/shimu/Downloads/DOGe-main/workbench/tasks/`
  任务历史、每天的执行流水、结果记录。
- `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/`
  支持 dashboard 的文件，以及共享的 Hopper 状态 / 任务解释。

## current 层规则

- `active_tasks.md`
  用来认领任务、记录状态、阻塞点、当前负责人。
- `latest_handoff.md`
  用来写当前最重要的一条交接摘要。
  这个文件要保持很短，只保留最新、最重要的方向信息。

谁负责更新 current 层：

- 谁改变了当前状态，谁就负责更新。
- 不要等待别的 AI 之后再补。

## 决策落盘规则

GUI 决策和 terminal 决策都不会自动落盘，必须手动写。

只要发生以下情况，就必须写一条短摘要到共享文件：

- 改了计划
- 选定或否定了一个方案
- 发现 blocker
- 拿到了关键结果
- 要交接给别的 AI
- 暂停了未完成工作
- 提交或修改了重要 Hopper 任务

写到哪里：

- 如果改变了当前方向：更新 `/Users/shimu/Downloads/DOGe-main/workbench/current/latest_handoff.md`
- 如果是具体工作事件或结果：追加到 `/Users/shimu/Downloads/DOGe-main/workbench/tasks/` 里的当天任务记录

不要把 GUI 对话内容或 terminal 输出本身当作共享记忆，只有写到磁盘上的摘要才算共享状态。

## 任务记录规则

- 默认只读 `/Users/shimu/Downloads/DOGe-main/workbench/tasks/LATEST.md`
- 只有在最新文件明确指向旧记录时，才继续读更早的任务文件
- 每天一个任务文件，命名格式为 `YYYY-MM-DD.md`
- 如果当天文件已经存在，就继续追加，不要新建重复文件
- `LATEST.md` 要保持简短和最新

## knowledge 规则

- 只有可复用经验才写入 `/Users/shimu/Downloads/DOGe-main/workbench/knowledge/`
- 一次性的状态信息不要放在 knowledge 里
- 每个 knowledge 文件顶部都要有简短摘要

## 数据登记规则

- 如果你采集、删除、移动、废弃、判定某组数据失效，必须更新：
  `/Users/shimu/Downloads/DOGe-main/workbench/DATA_REGISTRY.md`
- 要记录发生了什么、数据当前或曾经存放在哪里、现在是什么状态。
- 除非数据登记表明确说明有效，不要默认某组数据仍然可用。
- 如果你的任务依赖数据可用性，行动前先读这个文件。

## dashboard 规则

- 如果你的工作会创建、更新、使用某个 Hopper reservation 下的任务，必须先读取并遵守：
  `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/task_record_prompt.md`
- 不要自创 task record 格式
- dashboard 的任务记录文件是：
  `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/task_records.md`
- dashboard 的 SSH 状态文件是：
  `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/ssh_status.md`

## OpenClaw 协同

- OpenClaw 可以唤醒其他 AI，但不能依赖 GUI 聊天记录
- 因此凡是需要让别的 AI 接手的关键信息，都必须写进上述共享文件

## 冲突避免

- 开始一个具体任务前，先检查 `active_tasks.md`
- 开工前先在里面认领
- 除非文件里明确写了允许并行，否则不要和别的 AI 同时做同一个具体任务

## 最低更新要求

当你完成了一段有意义的工作、准备结束当前会话前，请完成适用的更新：

- 确认你已经登记到 `AI_REGISTRY.md`
- 更新 `active_tasks.md`
- 如果当前方向有变化，更新 `latest_handoff.md`
- 追加当天任务记录
- 如果你改动了数据的生命周期或存储位置，更新 `DATA_REGISTRY.md`
- 如果涉及 Hopper reservation 任务，更新 dashboard 的 task record
