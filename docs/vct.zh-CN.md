# VirtualCellTool：运行与独立测试指南

[English](vct.md) | **简体中文** · [返回项目首页](../README.zh-CN.md)

**先用公开 PBMC3k 数据跑通工具，再用适合研究问题的疾病数据开展分析。** VCT 与基因筛选管线独立，不参与遗传证据评分；PBMC 演示不是 WHS 实验，也不能证明 SINEUP 实现了蛋白恢复或功能救援。

| 你要完成的事 | 对应章节 |
|---|---|
| 在新目录、新环境中运行 | [1. 安装](#1-安装独立环境) → [2. 模型](#2-下载模型) → [3. 数据](#3-准备-pbmc-演示数据) → [4. 启动](#4-启动工具) |
| 逐项检查网页功能 | [5. 手动核对](#5-逐项核对网页功能) |
| 运行自动化检查 | [6. 测试命令](#6-运行自动化检查) |
| 理解额外功能需要什么 | [7. 可选数据与功能](#7-可选数据与功能) |
| 为 Wiki 准备可信的案例 | [8. 记录结果](#8-为-wiki-保留测试结果) |

## 1. 安装独立环境

准备好 **Git 和 Conda**。macOS/Linux 使用终端，Windows 使用 Miniforge Prompt。`#` 开头的行是说明；Windows 的 cmd.exe 不识别这种注释，请只复制命令行本身。

独立测试建议从远程克隆到一个**尚不存在的新目录**。如果已有专门的干净测试目录，可跳过克隆并进入该目录：

```text
# 将远程代码克隆到独立测试目录
git clone https://github.com/panxuan-pku/pekinghsc-2026iGEM-targetscreening.git vct-independent-test

# 进入新仓库根目录；后续命令均在这里执行
cd vct-independent-test

# 在本次克隆中创建 VCT 专用环境
conda env create --prefix ./workspace/envs/virtual-cell --file vct/environment.yml

# 激活环境
conda activate ./workspace/envs/virtual-cell

# 检查已安装包的依赖是否兼容
python -m pip check
```

**预期输出：** `No broken requirements found.` 环境包含 Scanpy、PyTorch、SIGnature 和网页依赖，无需先安装筛选或 CNV 环境。SIGnature 从固定的上游提交安装，不需要复制旧项目中的第三方源码。

确认解释器和数据路径属于本次测试，避免沿用以前设置的环境变量：

```text
# 显示当前 Python、工作目录、数据目录和模型目录
python -c "import sys; from vct.src.paths import WORKSPACE, DATA_DIR, MODEL_DIR; print('Python:', sys.executable); print('Workspace:', WORKSPACE); print('Data:', DATA_DIR); print('Model:', MODEL_DIR)"
```

Python 应位于新仓库的 `workspace/envs/virtual-cell/` 下；其他路径应位于 `workspace/vct/` 下。如指向旧目录，先检查并清除旧的 `VCT_WORKSPACE`、`VCT_DATA_DIR`、`VCT_MODEL_DIR` 或 `SIGNATURE_DIR` 设置。后续使用新下载的模型、新生成的数据，不复制原项目的缓存或结果。

## 2. 下载模型

```text
# 下载、校验并解压官方 SCimilarity 模型文件
python -m vct.prepare_model
```

**预期输出：** `[OK] Model verified and extracted`。

脚本下载约 **118 MB** 的 [SIGnature 官方模型压缩包](https://zenodo.org/records/17903196)，核对文件大小和发布的 MD5，然后将 SCimilarity 文件解压到 `workspace/vct/models/model_files/scimilarity/`。来源和 SHA-256 保存于 `workspace/vct/models/source.json`。

已有 `workspace/vct/models/` 时程序会拒绝覆盖。独立复测请使用新的克隆目录；模型已成功准备后无需重复执行此步骤。

## 3. 准备 PBMC 演示数据

```text
# 下载 PBMC3k，重新计算嵌入、UMAP 和六个计算分组，保存到网页默认数据目录
python vct/src/prepare_data.py --computed-groups 6 --output workspace/vct/data
```

首次运行通过 Scanpy 下载约 **6 MB** 的 PBMC3k 原始矩阵，然后对齐模型基因空间、计算嵌入和 UMAP。**预期看到：**

- 原始数据包含 **2,700 个细胞**。
- 嵌入形状为 **`(2700, 128)`**，UMAP 形状为 **`(2700, 2)`**。
- 最后显示 `数据管线完成` 和 `workspace/vct/data` 的路径。

该目录包含七个文件：

| 文件 | 用途 |
|---|---|
| `pbmc3k_raw.h5ad` | 原始矩阵 |
| `expr_aligned.npz` | 对齐并标准化后的表达矩阵 |
| `embeddings.npy` | SCimilarity 嵌入 |
| `umap.npy` | 二维显示坐标 |
| `meta.csv` | 细胞编号及分组 |
| `gene_order.txt` | 与表达矩阵对应的基因顺序 |
| `pbmc_info.json` | 数据规模、输入校验值、模型位置和分组方法 |

**Computed group 1–6 是计算分组，不是已确认的细胞类型。** 它们来自同一批嵌入上的固定随机种子 KMeans；不能直接改称 B 细胞、T 细胞等。

<details>
<summary>已有下载文件、已核实标签，或需要另存数据</summary>

使用独立下载的原始 PBMC 文件时，在上述命令后添加 `--input path/to/pbmc3k_raw.h5ad`。

如有来源明确的细胞标签，将 `--computed-groups 6` 替换为 `--metadata labels.csv`。该 CSV 必须包含唯一的 `cell` 编号和非空的 `cell_type` 标签，且恰好覆盖输入矩阵中的全部细胞。不要沿用来源无法核实的历史标签。

`--output` 必须指向尚不存在的目录。若改用其他输出目录，启动服务前还需设置 `VCT_DATA_DIR` 指向它，否则网页仍读取默认的 `workspace/vct/data/`。在启动服务的同一终端中，按所用 shell 选择一种：

```text
# macOS/Linux：使用另一次数据准备的输出目录
export VCT_DATA_DIR="$PWD/workspace/vct/data_02"
```

```text
# Windows PowerShell：使用另一次数据准备的输出目录
$env:VCT_DATA_DIR = "$PWD/workspace/vct/data_02"
```

```text
# Windows cmd.exe：仅复制下一行命令
set "VCT_DATA_DIR=%CD%\workspace\vct\data_02"
```

也可通过 `VCT_WORKSPACE` 调整整个 VCT 工作目录，或通过 `VCT_MODEL_DIR` 指定模型目录。`SIGNATURE_DIR` 仅用于显式选择第三方源码目录；常规安装使用环境中已安装的包即可。

</details>

## 4. 启动工具

```text
# 启动本地服务；保持当前终端运行
python -m uvicorn web.app:app --app-dir vct --host 127.0.0.1 --port 8377
```

看到 **`Application startup complete`** 后，在浏览器打开 **http://127.0.0.1:8377/**。选择顶部的 **PBMC 2700**。首次加载需要读入数据并拟合模型，等待页面显示细胞数量和 UMAP。

停止服务时，在原终端按 **Ctrl+C**。重新启动只需激活环境并执行启动命令，无需重新下载模型或生成数据。这是本地应用：其他使用者需要克隆并在自己的电脑上运行；`127.0.0.1` 不是公开托管地址。

<details>
<summary>页面无法打开、端口占用或数据加载失败</summary>

先查看服务终端的错误。端口占用时可将启动命令中的 `--port 8377` 改为 `--port 8378`，浏览器地址也改为 `http://127.0.0.1:8378/`。第 6 节接口测试默认连接 8377，改端口时需要同步设置 `VCT_PORT`。

数据加载失败时，核对第 1 节输出的路径及第 3 节的七个文件。若只准备了 PBMC，切换到 MS 或 Williams 会因缺少数据而无法加载；先切回 PBMC。健康接口 `http://127.0.0.1:8377/api/health` 只能帮助确认服务是否响应，不能替代数据与功能测试。

</details>

## 5. 逐项核对网页功能

建议先在 UMAP 中选择一个细胞，再搜索 **MS4A1**。每一步核对当前数据集、细胞编号和基因，结果会显示在右侧标签页中。

| 操作 | 应看到什么 | 结果表示什么 |
|---|---|---|
| 选择 PBMC，查看 UMAP 和分组图例 | 2,700 个细胞，Computed group 1–6 | 表达嵌入与计算分组 |
| 点击 UMAP 中一个细胞，再点击“归因结果出图” | 细胞编号、归因基因表及归因/表达对照图 | 输入基因对编码器表示的影响，不是疾病致病性排名 |
| 搜索并选中 MS4A1，点击“看表达分布” | 按表达量着色的 UMAP 和分组统计 | 观测表达，不是扰动后实测数据 |
| 点击 CIPHER“敲低”，再点击“过表达” | 两种方向对应的变化图和分组响应摘要 | 当前数据集整体拟合的线性预测 |
| 回到 MS4A1 的敲低结果，点击“对比线性基线” | CIPHER 与基线对照图 | 检查预测相对于简单基线的差异；差异本身不证明准确 |
| 展开“编码器敏感性探索”，调整目标表达量并查看单基因位移 | 原始与修改后编码的位置或距离 | 仅修改输入后重新编码，不是实际细胞迁移或类型转化 |
| 在同一折叠区比较 Top1、Top3、Top5 | 联合修改归因排名靠前基因后的编码结果 | 编码器敏感性，不是多靶点治疗效果 |

**预测边界：** CIPHER 只对当前数据集所选的高变基因提供预测；能在搜索框找到某基因，不等于该基因可被 CIPHER 或 GEARS 预测。不覆盖时应有明确提示，不能将缺失结果记为零效应。

**读图边界：** 网页“过表达→2×”指当前模型处理后的表达空间中将目标值设为对照均值的两倍，不等于蛋白量翻倍。分组摘要是同一整体预测与各组表达特征的对应关系，**不是各细胞类型分别拟合的因果预测**。界面中的“最敏感”或变化基因计数也不能作为实验验证或统计显著性的证据。

PBMC 核心测试不要求 GEARS 可用；其按钮或提示不能代替独立的 GEARS 测试。历史资料缺失也不应阻止上述操作。

## 6. 运行自动化检查

### 6.1 离线测试

在仓库根目录、VCT 环境中运行。无需启动网页服务：

```text
# 安装可选测试依赖；这里不会安装筛选环境或筛选数据
python -m pip install -r screening/requirements-dev.txt

# 检查 VCT 的核心逻辑、输入校验和网页相关行为
python -m pytest tests/vct -q
```

迁移版本曾得到 **68 passed、1 skipped、227 subtests passed**。其中一个测试因缺少可选历史资料而跳过。测试数量可能随代码变化；检查是否有 `failed` 或 `error`，并阅读跳过原因。离线测试包含合成数据用例，不能替代真实网页操作。

### 6.2 真实服务接口测试

保持第 4 节服务运行，在**另一个终端**进入同一测试仓库并激活同一 VCT 环境：

```text
# 激活本次克隆的 VCT 环境
conda activate ./workspace/envs/virtual-cell

# 对真实 PBMC 服务测试核心接口，明确排除尚未准备的 GEARS
python tests/vct/tests_api.py --datasets pbmc --skip-gears
```

预期显示 `pbmc: 2700 cells`、MS4A1 的非零 `|Δ|max`，以及：

- `17/19 API routes and home page passed`。
- `4 error contracts and 1 dataset contracts passed`。
- `SKIPPED: gene_coverage / perturb_causal (GEARS); not a full API pass`。

这里的 17/19 是**该脚本列出的检查范围**，不是全部功能均通过；它跳过 GEARS，也未测试 MS/Williams 数据集。该脚本针对固定演示规模编写，不适合直接验收任意自定义数据集。接口通过后仍需完成第 5 节的浏览器检查，确认图表和按钮实际可用。

如果服务改用了 8378 端口，在测试终端中设置 `VCT_PORT`：macOS/Linux 用 `export VCT_PORT=8378`，PowerShell 用 `$env:VCT_PORT = "8378"`，cmd.exe 用 `set VCT_PORT=8378`，然后再运行测试命令。

## 7. 可选数据与功能

**PBMC 流程通过只代表核心演示可运行，以下功能需另行验证。**

| 功能 | 额外条件 | 本次迁移的验证范围 |
|---|---|---|
| MS / Williams 数据集 | 原始矩阵、必要元数据及对应准备流程 | 未重新运行大型疾病数据集 |
| GEARS | Norman Perturb-seq 数据和兼容检查点；目标须在模型覆盖范围内 | 未重新训练或验证预训练预测 |
| 历史实验页面 | 明确列入白名单的本地历史文件 | 新克隆中允许显示缺失 |
| 自定义数据 | 兼容的基因标识、表达矩阵和细胞注释 | 不能假定任意 h5ad 都兼容 |

<details>
<summary>MS、Williams、GEARS 和历史资料的准备入口</summary>

Williams 准备脚本接收显式输入路径，要求包含 `obs["condition"]`：

```text
# 用实际 Williams 数据路径替换占位路径，准备网页数据
python vct/src/prepare_ws_web.py --input path/to/annotated_williams.h5ad
```

MS 网页准备脚本要求 `var["feature_name"]`、`obs["disease"]`，以及用同一矩阵、同一细胞顺序生成的 `ms_embeddings.npy`。这些嵌入由 `compute_ms_attribution.py` 生成；该脚本还要求供体和病灶元数据。将该次输出的嵌入放到 `VCT_DATA_DIR`，再运行 `prepare_ms_web.py --input ...`。这不是自动下载流程。使用前查看各脚本的 `--help` 和输入要求，保留数据来源、版本及注释依据，参见[来源说明](ATTRIBUTIONS.md)。

GEARS 使用独立于 PBMC 的训练数据和模型，参见 [GEARS 数据说明](https://github.com/snap-stanford/GEARS)及[检查点来源](https://huggingface.co/matthewshu/gears-norman)。默认位置为 `workspace/vct/gears_data/` 和 `workspace/vct/gears_ckpt/`。完成安装和资产准备后可检查加载：

```text
# 仅在 GEARS 数据与检查点均准备好之后运行
python vct/src/load_gears.py
```

此命令不代表模型已在 WHS 或其他疾病中得到验证。GEARS 缺失不影响 PBMC 归因和 CIPHER 功能。

`vct/analysis/` 保留研究分析脚本。`vct/src/history_panel.py` 只提供白名单中的本地历史文件；新克隆显示缺失属于预期情况。既往实验见[原项目](https://github.com/panxuan-pku/igem-backup)，无需复制历史材料来证明核心工具可用。

</details>

当前安装、PBMC 准备和核心服务实际验证于 **macOS Apple Silicon**。Windows/Linux 的命令形式已提供，但迁移时未在这两个系统上执行验证。

## 8. 为 Wiki 保留测试结果

记录**提交编号、操作系统、环境、数据来源、关键操作与通过/失败情况**。保留本次 `pbmc_info.json` 和 `models/source.json`，不要仅凭截图判断数据来源一致。

```text
# 记录本次测试所用的确切代码版本
git rev-parse HEAD

# 记录是否存在可能影响结果的本地修改
git status --short
```

建议保存四组截图：**环境和数据准备成功、PBMC UMAP、MS4A1 表达与归因、CIPHER 敲低/过表达及基线对照**。截图中保留数据集、基因、方向和细胞编号等必要上下文；本地工作区可用于保存记录，数据和结果无需提交到源码仓库。

| Wiki 板块 | 可以依据本次测试更新的内容 |
|---|---|
| Documents | 独立安装、模型下载、数据准备、启动步骤和预期输出 |
| Tools | 最小 PBMC 演示、实际操作图及完整版仓库入口 |
| Model | 实际模型输入输出、基因覆盖和分组解释边界 |
| Engineering | 经记录的实验与迭代结论；不能把运行成功写成疾病机制验证 |

用于 Wiki 的图片或视频应按团队要求上传到 iGEM 静态资源平台。**PBMC 测试可证明工具演示流程可运行，不能支撑 WHS 特异性结论。** 完成测试并确认问题后，再单独更新 Wiki 正文。
