# iGEM 2026：微缺失候选基因优先级筛选

[English](README.md) | **简体中文**

**输入缺失区间或候选基因列表，获得基于证据的排名和设计条件检查报告。** 管线整合 ClinGen 和 gnomAD 证据，补充 Open Targets（OT）疾病注释，并检查本项目的 **3′UTR ≥30 bp** 和模块长度预算要求。

| 你的起点 | 对应流程 |
|---|---|
| 已知缺失区间，或第一次运行 WHS 示例 | [2. 缺失区间](#2-缺失区间whs-示例) |
| 已有来自样本、文献或分析的候选基因列表 | [3. 基因列表](#3-基因列表) |
| 查看结果或自定义输出位置 | [4. 输出结果](#4-输出结果) |
| 了解项目文件的组织方式 | [7. 仓库结构](#7-仓库结构) |

独立的 **VirtualCellTool（VCT）** 使用单细胞数据探索表达变化。它不参与遗传证据排名，也不能证明 SINEUP 已实现蛋白水平恢复。[开始 VCT 演示与独立测试](docs/vct.zh-CN.md)。

## 1. 安装环境

macOS/Linux 使用终端；Windows 使用已启用 Conda 的终端，例如 Miniforge Prompt。**请先安装 Git 和 Conda。** 以下命令适用于这三个系统。以 `#` 开头的行用于解释命令；使用 Windows Miniforge Prompt（cmd.exe）时，只复制命令行本身：

```text
# 从 GitHub 下载项目
git clone https://github.com/panxuan-pku/pekinghsc-2026iGEM-targetscreening.git

# 进入仓库根目录
cd pekinghsc-2026iGEM-targetscreening

# 创建独立的 Conda 环境，并安装基础筛选依赖
conda env create --prefix ./workspace/envs/virtual-screening --file screening/environment.yml

# 在当前终端激活筛选环境
conda activate ./workspace/envs/virtual-screening

# 检查依赖声明、已安装版本及模块导入是否正常
python -m screening.check --suite environment
```

看到 **`[OK] Checks passed (environment)`** 后继续。

## 2. 缺失区间：WHS 示例

本例使用 **ClinGen 的 Wolf–Hirschhorn 综合征（WHS）参考区间 ISCA-37429，基因组版本为 GRCh38**。管线从参考注释中提取与区间重叠的蛋白编码基因，无需从 BioMart 导出。这一公开参考区间并非某位患者的实际缺失区间，也不等同于历史研究中的 19 基因集合。

### 2.1 准备输入

创建 `workspace/screening/input/raw/` 文件夹，将 [ClinGen GRCh38 区间表](https://ftp.clinicalgenome.org/ClinGen_region_curation_list_GRCh38.tsv) 下载到其中，保留原文件名。可通过 [WHS 记录](https://search.clinicalgenome.org/kb/gene-dosage/region/ISCA-37429)核对区间。

```text
# 将 ClinGen 的 WHS 区间记录转换为区间输入 JSON
python -m screening.prepare_input interval --input workspace/screening/input/raw/ClinGen_region_curation_list_GRCh38.tsv --format clingen-tsv --region-id ISCA-37429 --genome-build GRCh38 --source "https://search.clinicalgenome.org/kb/gene-dosage/region/ISCA-37429" --output workspace/screening/input/whs_interval_input.json
```

**预期输出：** `[OK] Input preparation complete`，以及区间坐标和 `whs_interval_input.json` 的路径。该记录的坐标为 **chr4:337779–2009235**，从 1 开始计数且包含两端。核对坐标即可，无需手动编辑 JSON。

### 2.2 准备参考数据

```text
# 下载五份参考文件；已有文件通过校验后直接复用
python -m screening.run prepare
```

**预期输出：** `[OK] Reference preparation complete (5/5)`。首次下载约 **617 MiB**，保存在 `workspace/screening/references/`。后续运行会校验并复用已完成的文件；下载中断后重新执行相同命令即可。

| 参考数据 | 用途 |
|---|---|
| GENCODE 和 HGNC | 基因身份、位置和转录本注释 |
| ClinGen 和 gnomAD | 核心排名证据 |
| HPA | 组织表达注释 |

步骤 2.1 的 ClinGen **区间表**用于确定输入区间；本步骤下载的 **基因证据表**用于评分。

### 2.3 执行筛选

```text
# 对 WHS 区间内的基因进行排名，将报告保存到 whs_interval_01
python -m screening.run run --input-file workspace/screening/input/whs_interval_input.json --output workspace/screening/results/whs_interval_01
```

**预期输出：** `[OK] Screening complete`，以及候选数量和 **`interval_report.html`** 的路径。用浏览器打开该文件查看报告。使用 2026 年 9 月 28 日核验的参考数据，结果为：**30 个候选基因，14 个入选设计候选，NSD2 排名第一**。在线证据更新可能使结果发生变化。其他输出见[第 4 节](#4-输出结果)。

<details>
<summary>使用自己的缺失区间：BED 输入</summary>

将 GRCh38 BED 文件保存为 `workspace/screening/input/raw/deletion.bed`，用以下命令替换步骤 2.1。将 `actual interval source` 替换为实际区间来源：

```text
# 将自己的 GRCh38 BED 区间转换为输入 JSON；请替换来源占位文字
python -m screening.prepare_input interval --input workspace/screening/input/raw/deletion.bed --format bed --genome-build GRCh38 --source "actual interval source" --output workspace/screening/input/deletion_interval_input.json
```

步骤 2.2 不变。步骤 2.3 改用 `--input-file workspace/screening/input/deletion_interval_input.json`，并指定新的输出目录。转换器将 BED 从 0 开始的起点转换为从 1 开始、包含两端的坐标；**不进行基因组版本转换**。多行 BED 可用 `--row` 选择一行。CSV/TSV 区间表的用法见 `python -m screening.prepare_input interval --help`。

</details>

## 3. 基因列表

如果你**已有明确的候选基因集合**，可以使用此流程，无需提供缺失坐标。支持 Ensembl/HGNC 编号和可唯一解析的基因符号。

### 3.1 准备输入

将表格保存为 `workspace/screening/input/raw/genes.csv`，每行一个候选，保留表头和其他列。以下示例假设基因所在列名为 **`gene_id`**；请将 `--gene-column` 和 `--source` 分别替换为实际列名及来源网址或文献编号。

```text
# 从 gene_id 列读取候选基因编号；请按实际情况替换列名和来源
python -m screening.prepare_input genes --input workspace/screening/input/raw/genes.csv --format csv --gene-column gene_id --source "actual source URL or publication ID" --output workspace/screening/input/gene_list_input.json
```

**预期输出：** `[OK] Input preparation complete`，以及基因标识符数量和 `gene_list_input.json` 的路径。基因身份将在筛选阶段核验；重复或无法唯一确定身份时，程序会明确报错。

<details>
<summary>其他格式：TXT、TSV 和 Excel</summary>

TXT 文件每行一个基因，不包含表头。将 `actual list source` 替换为列表的实际来源：

```text
# 将每行一个基因的 TXT 文件转换为输入 JSON；请替换来源占位文字
python -m screening.prepare_input genes --input workspace/screening/input/raw/genes.txt --format txt --source "actual list source" --output workspace/screening/input/gene_list_input.json
```

TSV 沿用 CSV 命令，改为实际 `.tsv` 路径、`--format tsv` 和实际列名。Excel 表格请先导出为 UTF-8 CSV。选择一种格式生成输入 JSON 即可。

</details>

<details>
<summary>还没有基因列表？用第 2 节的完整 WHS 候选集合练习</summary>

完成第 2 节后，使用以下命令替代上面的 CSV 转换命令。它读取**全部候选**，而非仅排名靠前的基因：

```text
# 读取区间筛选中的全部 WHS 候选，生成基因列表输入
python -m screening.prepare_input genes --input workspace/screening/results/whs_interval_01/interval_candidates.csv --format csv --gene-column gene_id --source "WHS ISCA-37429; whs_interval_01/interval_candidates.csv; see whs_interval_01/interval_manifest.json" --output workspace/screening/input/gene_list_input.json
```

接着执行步骤 3.2 和 3.3。这是在同一候选集合上演示另一种输入方式，不构成独立的候选来源。

</details>

### 3.2 准备参考数据

```text
# 校验并复用两种流程共享的参考数据，下载尚缺少的文件
python -m screening.run prepare
```

**预期输出：** `[OK] Reference preparation complete (5/5)`。两种输入方式共享参考数据，会复用步骤 2.2 中已校验的文件。

### 3.3 执行筛选

```text
# 对提交的基因列表进行排名，将报告保存到 gene_list_01
python -m screening.run run --input-file workspace/screening/input/gene_list_input.json --output workspace/screening/results/gene_list_01
```

**预期输出：** `[OK] Screening complete`，以及候选数量和 **`gene_list_report.html`** 的路径。报告覆盖你提交的基因列表。

两种流程默认使用 WHS 疾病编号 `MONDO_0008684` 获取 OT 注释。研究其他疾病时，在 `run` 命令中添加 `--disease-id YOUR_DISEASE_ID`，替换为对应疾病编号。OT 注释不改变核心排名。

## 4. 输出结果

### 4.1 应该查看哪些文件

文件名遵循 **`输入类型_用途.扩展名`**：缺失区间使用 `interval`，基因列表使用 `gene_list`。

| 用途 | 区间输入 | 基因列表输入 |
|---|---|---|
| 浏览器报告 | `interval_report.html` | `gene_list_report.html` |
| 摘要表，共 16 列 | `interval_summary.csv` | `gene_list_summary.csv` |
| 完整证据表 | `interval_ranked.csv` | `gene_list_ranked.csv` |
| 标准化候选列表 | `interval_candidates.csv` | `gene_list_candidates.csv` |
| 运行清单 | `interval_manifest.json` | `gene_list_manifest.json` |
| 运行配置 | `interval_config.yaml` | `gene_list_config.yaml` |
| OT 响应快照 | `interval_ot_snapshot.json` | `gene_list_ot_snapshot.json` |

CSV 使用带 BOM 的 UTF-8 编码，便于 Excel 打开；JSON、HTML 和 YAML 使用 UTF-8。运行清单记录输入类型、输出文件名和校验值。共享参考数据保存在 `workspace/screening/references/`。

摘要表和完整证据表保留相同的候选、排名及分数。**分数表示研究优先级；`selected` 表示满足设计长度和预算条件。** 二者均不代表已通过实验验证。

**`[WARN] Ranking generated…`** 表示排名已生成，但 OT 注释有缺失，请查看报告。**`[FAIL]`** 表示当前步骤已停止，需解决提示的问题后重试。

### 4.2 自定义输出目录

在 `run` 命令中设置 **`--output "directory"`**，其中 `directory` 为目标路径：

```text
# 将本次结果保存到自定义目录；该目录必须尚不存在
python -m screening.run run --input-file workspace/screening/input/gene_list_input.json --output "workspace/screening/results/gene_list_02"
```

| 系统 | 绝对路径示例 |
|---|---|
| macOS | `--output "/Users/yourname/Documents/screening/gene_list_02"` |
| Linux | `--output "/home/yourname/screening/gene_list_02"` |
| Windows | `--output "D:\screening\gene_list_02"` |

相对路径以终端当前目录为基准。**输出目录必须尚不存在**，程序会自动创建。再次运行时，可将 `_01` 改为 `_02`。

不指定 `--output` 时，程序会在仓库内自动创建 `workspace/screening/results/interval_YYYYMMDD_HHMMSS_microseconds/` 或 `gene_list_YYYYMMDD_HHMMSS_microseconds/`。自定义目录名会原样使用，其中的文件仍保留输入类型前缀。建议将准备好的输入命名为 `project_interval_input.json` 或 `project_gene_list_input.json`。已有结果不会被重命名。

如需更改参考数据位置，请在 `prepare` 和 `run` 中使用相同的 `--references "directory"`。

## 5. 方法与复现

统一配置文件为 **`screening/config/screening.yaml`**。ClinGen 和 gnomAD 用于排名；OT、HPA 和可选的 IMPC 用于注释。3′UTR 检查和模块预算决定设计候选是否入选，但不删除排名中的基因。当前配置未启用 DeepLOF。

`references.json` 记录参考数据的版本、来源和校验值。复现分析时，保留输入、参考数据及配置，并在 `run` 命令中添加 `--ot-snapshot previous_run/interval_ot_snapshot.json`，指向此前的 OT 快照；基因列表流程使用 `gene_list_ot_snapshot.json`。

证据的具体作用见[方法与溯源说明（英文）](docs/methods.md)，下游探索见 [VCT 中文指南](docs/vct.zh-CN.md)。两种工具均不能证明治疗效果。

## 6. 可选功能与开发测试

基础筛选只需安装第 1 节的环境。VCT 使用**独立环境**，其中已包含所需的单细胞依赖，无需先安装 CNV 扩展包。

| 用途 | 操作说明 |
|---|---|
| VCT：安装、模型、数据和交互演示 | [VCT 中文指南](docs/vct.zh-CN.md) |
| 可选单细胞/CNV 处理 | 在筛选环境中运行 `python -m pip install -r screening/requirements-cnv.txt`；核心流程不需要 |
| 开发测试 | 执行下列命令；普通使用者无需安装 |

```text
# 安装可选的开发测试依赖
python -m pip install -r screening/requirements-dev.txt

# 运行核心筛选测试，不下载参考数据
python -m screening.check --suite core

# 检查发布文件是否夹带数据或模型，以及本地文档链接是否有效
python tests/check_release.py
```

预期看到 **`[OK] Checks passed (core)`** 和 **`Release check passed`**。VCT 测试需按其指南在 VCT 环境中运行。可选 CNV 测试需要先安装 CNV 依赖，再执行 `python -m screening.check --suite full`。仓库保留了 CNV 探索功能，但尚未验证它能够从 scRNA-seq 数据中推断出本例的 WHS 候选集合。

## 7. 仓库结构

```text
pekinghsc-2026iGEM-targetscreening/  # 仓库根目录
├── README.md                 # 英文入口：两种筛选输入流程
├── README.zh-CN.md           # 简体中文入口
├── screening/                # 基因优先级筛选管线
│   ├── prepare_input.py      # 下载的表格 → 带输入类型的 JSON
│   ├── run.py                # 参考数据准备和筛选
│   ├── report.py             # 英文 HTML 报告
│   ├── check.py              # 环境检查和开发测试
│   ├── config/               # 当前流程使用 screening.yaml
│   ├── src/                  # 证据整合、排名和可选分析
│   └── environment.yml       # 基础筛选 Conda 环境
├── vct/                      # 虚拟细胞工具
│   ├── src/                  # 数据准备和扰动引擎
│   ├── web/                  # 本地交互应用
│   ├── analysis/             # 可选研究分析
│   ├── prepare_model.py      # 下载并校验官方模型
│   └── environment.yml       # 独立 VCT Conda 环境
├── tests/                    # 筛选、VCT 和发布检查
├── docs/                     # 方法、VCT 指南和第三方来源说明
├── examples/                 # 示例来源和复现说明
├── wiki_figures/             # 绘图源码和本地待上传图片
│   ├── generate_model.py     # 总览、已核验的 WHS 得分贡献、设计检查
│   ├── model/               # 本地 SVG/PNG 图片及数据来源记录
│   ├── document/            # 预留：使用指南配图
│   ├── tool/                # 预留：工具配图
│   └── engineering/         # 预留：DBTL 配图
├── .archify/                 # 本地生成的代码流程图及审阅产物
└── workspace/                # 仅本地保存：环境、数据、模型和结果
```

仓库不包含历史疾病研究目录或预置结果。`workspace/` 在本地创建，并被 Git 忽略。较早的 `screening/config/pipeline.yaml` 和 CNV 配置用于可选分析及回归测试；**上述流程通过 `python -m screening.run` 使用 `screening/config/screening.yaml`**。

Wiki 配图见[绘图说明](wiki_figures/README.md)。绘图源码和目录占位文件可纳入版本管理；生成的 SVG/PNG、预览页和运行来源记录仅保留在本地，并被 Git 忽略。确认图片后，上传到 iGEM 静态资源平台，再由 Wiki 引用。`.archify/` 保存本地生成的交互式代码流程图及审阅产物，同样被 Git 忽略，不是程序运行依赖。Wiki 单独维护，这两个目录都不会自动更新网页正文。

## 许可证与来源

团队代码采用 [MIT 许可证](LICENSE)。第三方软件、数据和模型来源见[来源说明（英文）](docs/ATTRIBUTIONS.md)，其各自的使用条款仍然适用。
