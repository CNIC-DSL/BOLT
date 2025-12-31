# 架构与执行流程

本文面向二次开发者，说明 bolt-lab 的核心执行链路、关键数据结构、以及各模块之间的责任边界。

一、整体数据流（从 YAML 到结果汇总）

1）入口：bolt-grid
- CLI 接收参数：--config、--output-dir、--model-dir
- 将配置文件定位到实际路径后，交给 runner 执行（runner 会读取 YAML 并启动调度）

2）runner：run_grid.py
- 读取 YAML 并校验必需字段：
  maps / methods / datasets / grid / run / paths / result_file / per_method_extra
- 调用 set_paths(results_dir, logs_dir, result_file) 初始化：
  results/summary_{result_file}.csv
  results/seen_index_{result_file}.json
  logs/ 根目录
- 将 grid 中每个维度展开为组合（combo），形成任务列表 combos
- 建立 GPU token 池（Queue）：
  - run.gpus 指定可用 GPU 列表
  - run.slots_per_gpu 决定每张 GPU 提供多少个 token
  - token 总数 = len(gpus) * slots_per_gpu
- 使用 ThreadPoolExecutor 并发调度：
  - 每个 worker 启动时领取一个 gpu_id
  - 执行完毕归还 gpu_id
  - 并发度 = min(max_workers, token 总数, combos 数)

3）单组合执行：utils.run_combo()
- 根据 method 在两个 registry 中查找方法定义：
  - cli_gcd.METHOD_REGISTRY_GCD
  - cli_openset.METHOD_REGISTRY_OPENSET
- 生成 args_json（make_base_args）：
  包含 method/dataset/ratio/fold/seed/gpu/epochs 等参数，并合并 per_method_extra 中该 method 的扩展字段
- 执行 stages：
  - 每个 stage 由 cli_builder(args_json, stage_idx) 生成命令行数组
  - run_stage() 负责：
    - 设置 ARGS_JSON 环境变量（用于记录与复现）
    - 设置 CUDA_VISIBLE_DEVICES（绑定到领取到的 gpu_id）
    - 将 CMD 与 ARGS_JSON 记录到 stage 日志开头
    - 启动子进程并等待返回码
  - 任一 stage 返回码非 0 则该 combo 失败，后续阶段不会继续

4）结果收集与汇总：collect_latest_result() + write_summary()
- combo 全部 stage 成功后，框架会尝试读取：
  ./results/{task}/{method}/results.csv
  并使用最后一行作为“最新结果”
- 若读取成功，则追加写入：
  results/summary_{result_file}.csv
- 若未找到结果文件或结果为空，会提示 Finished but no results found，不写 summary

二、关键模块与职责边界

1）run_grid.py（调度层）
- 负责：解析 YAML、展开组合、并发调度、OOM 重试策略（可选）
- 不负责：方法内部训练逻辑与指标计算

2）cli_gcd.py / cli_openset.py（方法注册与命令构建）
- 负责：将 method 映射到 stages，并为每个 stage 提供 cli_builder
- 负责：定义默认 config 路径与默认 output_base
- 不负责：执行过程中的结果写入（这由方法脚本完成）

3）utils.py（运行与汇总层）
- 负责：构造 args_json、运行 stage、组织日志目录、收集结果并写入 summary
- 负责：为每个 stage 写入 CMD 与 ARGS_JSON（用于复现与排查）

三、关于可复现性的约定

1）ARGS_JSON 是复现的最小闭环
- 每个 stage 日志开头都会包含：
  - CMD（实际执行命令）
  - ARGS_JSON（完整参数 JSON）
因此“复现实验”通常可以从日志中直接拿到命令与参数。

2）结果文件路径约定
- 自动汇总依赖以下约定：
  ./results/{task}/{method}/results.csv
方法脚本必须在运行结束时写入该文件，否则框架无法自动收集并汇总。

四、开发者常用调试手段

1）dry_run
- 在 YAML 的 run 中设置 dry_run=true，可只打印命令与 ARGS_JSON，不实际执行
- 适合检查组合数、参数传递与命令拼接是否符合预期

2）从 stage 日志复现
- 找到失败 combo 的 stageX.log
- 复制 CMD 与 ARGS_JSON，对照方法脚本的参数解析逻辑进行复现与定位
