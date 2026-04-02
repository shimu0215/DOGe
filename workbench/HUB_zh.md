# Workbench 总说明

这个 workbench 现在只保留最小共享系统。

## 对合作 AI 的要求

1. 在这里开始做有意义的工作前，先按这个格式给自己起一个唯一名字：
   `<ai名-丰缘地区宝可梦名>`
   例如：`codex-水跃鱼`
2. 把这个名字登记到 `/Users/shimu/Downloads/DOGe-main/workbench/AI_REGISTRY.md`
3. 之后所有面向用户的回复都先说自己的登记名字
4. 阅读并遵守 `/Users/shimu/Downloads/DOGe-main/workbench/CORE_INFO.md`
5. 遵守核心信息里的执行流程：
   所有代码必须先在本地修改、push 到仓库、再到 Hopper pull 后运行
6. 除非用户明确要求，任何情况下都不能取消 Hopper job；如果需要停止任务，只能 kill job 内部进程

## workbench 现在保留的内容

- `/Users/shimu/Downloads/DOGe-main/workbench/AI_REGISTRY.md`
  AI 名字登记表。
- `/Users/shimu/Downloads/DOGe-main/workbench/CORE_INFO.md`
  所有 AI 都要遵守的核心信息记录。
- `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/`
  本地 dashboard，只展示直接从 Hopper 获取的 reservation、running job、GPU 和 SSH 状态。
