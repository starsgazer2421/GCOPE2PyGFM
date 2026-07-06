# GCOPE 实验手册（PyGFM 集成版）

本目录提供 GCOPE 在 PyGFM 中的实验入口。GCOPE 原始源码被迁入 `src/pygfm/baseline_models/gcope/original_src/` ，并通过适配层 `src/pygfm/baseline_models/gcope/runner.py` 将 PyGFM 风格 YAML 配置转换为GCOPE 原始 `fastargs` 参数。

当前配置默认复现论文中的 Cora 1-shot 迁移实验：预训练时将 Cora 作为目标数据集留出，使用其余 9 个数据集作为 source datasets；微调时再把预训练模型迁移到 Cora。

## 安装

进入 `GCOPE2PyGFM` 项目根目录后安装依赖：

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install torch-geometric
pip install -e . --no-deps
pip install fastargs torchmetrics dgl tqdm pandas terminaltables networkx sympy
```

其中第一行安装 PyG 对应 PyTorch/CUDA 版本的预编译扩展包，避免在服务器上长时间源码编译；`pip install -e . --no-deps` 只安装本地 PyGFM 包本身，GCOPE 原始代码所需的 `fastargs`、`torchmetrics`、`dgl`、`networkx`、`sympy` 等依赖在最后一行手动安装。

## 实验流程

### Step 1  跨域预训练

```bash
python scripts/gcope/pretrain.py -c scripts/gcope/configs/pretrain.yaml
```

预期产物：

```text
storage/gcope/cora_gcope/wisconsin,texas,cornell,chameleon,squirrel,citeseer,pubmed,computers,photo_pretrained_model.pt
storage/gcope/cora_gcope/config.json
```

配置含义：

```text
stage: pretrain
method: GraphCL
target dataset left out: Cora
source datasets: Wisconsin, Texas, Cornell, Chameleon, Squirrel, Citeseer, Pubmed, Computers, Photo
cross_link: 1
reconstruct weight: 0.2
backbone: FAGCN
```

### Step 2  微调迁移

```bash
python scripts/gcope/finetune.py -c scripts/gcope/configs/finetune.yaml
```

预期产物：

```text
storage/gcope/cora_gcope/cora_results.txt
storage/gcope/cora_gcope/config.json
```

配置含义：

```text
stage: finetune
target dataset: Cora
few_shot: 1
repeat_times: 5
pretrained_file: storage/gcope/cora_gcope/wisconsin,texas,cornell,chameleon,squirrel,citeseer,pubmed,computers,photo_pretrained_model.pt
backbone_tuning: true
```

### Step 3  ProG 提示迁移

```bash
python scripts/gcope/prog.py -c scripts/gcope/configs/prog.yaml
```

预期产物：

```text
storage/gcope/cora_prog/cora_results.txt
storage/gcope/cora_prog/config.json
```

配置含义：

```text
stage: prog
target dataset: Cora
prompt model: HeavyPrompt
backbone_tuning: false
repeat_times: 5
```

### Step 4  端到端监督训练

```bash
python scripts/gcope/ete.py -c scripts/gcope/configs/ete.yaml
```

预期产物：

```text
storage/gcope/cora_ete/cora_results.txt
storage/gcope/cora_ete/results.pt
storage/gcope/cora_ete/config.json
```

配置含义：

```text
stage: ete
target dataset: Cora
training mode: end-to-end supervised baseline
repeat_times: 5
```

## 消融实验

本目录只保留 `pretrain.yaml`、`finetune.yaml`、`prog.yaml` 和 `ete.yaml` 四个主要配置文件。RQ2、RQ3 的消融实验不再单独保留额外 YAML，可以在主配置上临时修改关键参数后运行。

RQ2 关闭 coordinator 间连接边：

```text
pretrain.yaml:
  general.save_dir: storage/gcope/cora_gcope_no_inter_edges
  pretrain.cross_link_ablation: true

finetune.yaml:
  general.save_dir: storage/gcope/cora_gcope_no_inter_edges
  adapt.pretrained_file: storage/gcope/cora_gcope_no_inter_edges/wisconsin,texas,cornell,chameleon,squirrel,citeseer,pubmed,computers,photo_pretrained_model.pt
```

RQ3 不使用 reconstruction loss：

```text
pretrain.yaml:
  general.save_dir: storage/gcope/cora_gcope_rec0
  pretrain.reconstruct: 0.0

finetune.yaml:
  general.save_dir: storage/gcope/cora_gcope_rec0
  adapt.pretrained_file: storage/gcope/cora_gcope_rec0/wisconsin,texas,cornell,chameleon,squirrel,citeseer,pubmed,computers,photo_pretrained_model.pt
```

消融实验结束后，建议把 `pretrain.yaml` 和 `finetune.yaml` 改回默认配置，避免后续 RQ1/RQ4 复现实验混用消融参数。

## PyGFM 统一入口

除直接运行 `scripts/gcope/*.py` 外，也可以通过 PyGFM 统一 CLI 调用：

```bash
pygfm -c scripts/gcope/configs/pretrain.yaml
pygfm -c scripts/gcope/configs/finetune.yaml
pygfm -c scripts/gcope/configs/prog.yaml
pygfm -c scripts/gcope/configs/ete.yaml
```

## 更换目标数据集

如果要从 Cora 改成其他目标数据集，需要同步修改以下位置：

```text
pretrain.yaml:
  data.name: 删除目标数据集，只保留 source datasets
  general.save_dir: 建议改成对应目标数据集的输出目录

finetune.yaml / prog.yaml / ete.yaml:
  data.name: 改成新的目标数据集
  general.save_dir: 改成对应目标数据集的输出目录
  adapt.pretrained_file: 改成 pretrain 阶段实际生成的 checkpoint 路径
```

例如目标数据集改为 Photo 时，预训练 `data.name` 应该包含除 Photo 以外的 9 个数据集，微调 `data.name` 改为 `photo`，并把 `pretrained_file` 指向新的预训练模型文件。
