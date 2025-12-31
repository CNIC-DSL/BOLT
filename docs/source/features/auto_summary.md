# 自动结果收集与汇总

bolt-lab 在每个实验组合结束后，会自动收集该组合对应的结果记录，并追加写入统一的汇总表，便于后续对比与分析。

## 1. 结果文件约定

框架默认从以下位置读取单方法的结果文件：
./results/{task}/{method}/results.csv

其中：
- task 为 gcd 或 openset
- method 为方法名（例如 tan、plm_ood、ab 等）

框架会读取该 CSV 的最后一行作为“最新结果记录”。

说明：
- 如果 results.csv 不存在或为空，框架不会写入汇总表，并会打印提示信息。
- 因此，每个方法脚本需要保证在运行结束后写入上述位置，且包含关键指标列。

## 2. 汇总文件位置与命名

汇总文件写入到 YAML 的 paths.results_dir 下：
{results_dir}/summary_{result_file}.csv

同时会生成索引文件：
{results_dir}/seen_index_{result_file}.json

其中 result_file 来自 YAML 顶层字段 result_file。

## 3. 汇总表包含哪些字段

汇总表至少包含：
- method、dataset
- known_cls_ratio、labeled_ratio、cluster_num_factor、seed
- 指标列（随任务类型不同而变化）
- args：一份 JSON 字符串，记录本次运行的关键参数（用于复现与去重）

一般情况下：
- GCD 常见指标包括 ACC、H-Score、K-ACC、N-ACC、ARI、NMI 等
- OpenSet 常见指标包括 F1、K-F1、N-F1 等

## 4. 常见问题：为什么没有写入 summary

按以下顺序排查：
1) 子任务是否执行失败
   先查看 logs/ 下该组合的 stage 日志，确认是否有错误堆栈或非 0 返回码。
2) results.csv 是否生成
   确认 ./results/{task}/{method}/results.csv 是否存在，且 CSV 至少有一行结果数据。
3) 结果列是否完整
   若方法脚本写出的列与框架期望列差异较大，可能导致汇总缺失或指标为空。建议在方法侧统一列名，或在方法侧补齐汇总所需字段。

## 5. 使用建议

- 进行方法对比时，直接使用 summary_{result_file}.csv 作为分析入口。
- 若要区分多次大实验，建议通过改变 result_file 生成不同的汇总文件，避免相互覆盖。
