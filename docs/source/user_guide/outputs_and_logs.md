# 输出与日志定位

本页说明运行完成后应当在哪里查看日志、结果汇总与产出文件，并给出常见问题的排查顺序。

## 1. 关键输出目录（位于工作目录 --output-dir 下）

- logs/
  每个组合（combo）的运行日志，包含分阶段日志与总日志。排查错误优先查看此处。

- results/
  汇总与索引文件目录。通常包含：
  - summary_{result_file}.csv
  - seen_index_{result_file}.json

- outputs/
  训练/评测产物（模型、预测、缓存等），按方法与组合组织。

## 2. 日志目录组织方式

每个 combo 的日志目录通常按以下层级组织：

logs/{task}/{method}/{dataset}/
  kr{known_cls_ratio}/
  lr{labeled_ratio}/
  {fold_type}_{fold_num}_{fold_idx}/
  seed{seed}/

目录内通常包含：
- stage1.log、stage2.log...：各阶段 stdout/stderr
- all.log：该组合的关键事件记录（开始、每阶段返回码、结束）

stage 日志的开头通常会写入两段信息：
- CMD：实际执行命令
- ARGS_JSON：本次组合的完整参数（用于复现与排查）

## 3. 结果文件与汇总文件

3.1 单方法结果文件（results.csv）
框架默认从以下位置读取单方法结果：
results/{task}/{method}/results.csv

框架会读取该 CSV 的最后一行作为“最新结果记录”，并将其写入汇总表。

3.2 汇总文件（summary_{result_file}.csv）
汇总文件位于：
results/summary_{result_file}.csv

每当某个 combo 成功收集到结果，汇总表就会追加一行。

## 4. 常见问题排查顺序

问题 A：组合失败 / 运行报错
1) 先找到该组合对应的 stageX.log
2) 查看报错堆栈与关键参数（ARGS_JSON）
3) 结合 CMD 可直接复现单次运行进行定位

问题 B：运行结束但 summary 没有新增记录
1) 检查该组合是否有任何 stage 返回码非 0（失败会直接停止且不写 summary）
2) 检查 results/{task}/{method}/results.csv 是否存在且有内容
3) 检查方法写出的列是否完整（特别是指标列与 args 字段）

问题 C：输出目录结构不符合预期
1) 确认 --output-dir 是否指向运行根目录（不建议直接指向 outputs 子目录）
2) 确认 paths.results_dir / paths.logs_dir 是否为相对路径（相对路径会落在工作目录下）
