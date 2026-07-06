# GCOPE 实验手册（PyGFM 集成版）

本目录提供 GCOPE 在 PyGFM 中的实验入口。GCOPE 原始源码被迁入 `src/pygfm/baseline_models/gcope/original_src/` ，并通过适配层 `src/pygfm/baseline_models/gcope/runner.py` 将 PyGFM 风格 YAML 配置转换为GCOPE 原始 `fastargs` 参数。

## 安装

进入 `GCOPE2PyGFM` 项目根目录后安装依赖：

```bash
pip install -e ".[torch,gcope]"
```

其中 `torch` 包含 PyTorch / PyG 相关依赖，`gcope` 包含 GCOPE 原始代码所需的 `fastargs`、`torchmetrics`、`dgl`、`networkx`、`sympy` 等依赖。

## 实验流程

### Step 1  跨域预训练

```bash
python scripts/gcope/pretrain.py -c scripts/gcope/configs/pretrain_smoke.yaml
```

预期产物：

```text
storage/gcope/pretrain_smoke/cora,citeseer_pretrained_model.pt
storage/gcope/pretrain_smoke/config.json
```

配置含义：

```text
stage: pretrain
method: GraphCL
source datasets: Cora, Citeseer
cross_link: 1
```

### Step 2  微调迁移

```bash
python scripts/gcope/finetune.py -c scripts/gcope/configs/finetune_smoke.yaml
```

预期产物：

```text
storage/gcope/finetune_smoke/cora_results.txt
storage/gcope/finetune_smoke/config.json
```

配置含义：

```text
stage: finetune
target dataset: Cora
few_shot: 1
pretrained_file: storage/gcope/pretrain_smoke/cora,citeseer_pretrained_model.pt
```

### Step 3  ProG 提示迁移

```bash
python scripts/gcope/prog.py -c scripts/gcope/configs/prog_smoke.yaml
```

预期产物：

```text
storage/gcope/prog_smoke/cora_results.txt
storage/gcope/prog_smoke/config.json
```

配置含义：

```text
stage: prog
target dataset: Cora
prompt model: HeavyPrompt
backbone_tuning: false
```

### Step 4  端到端监督训练

```bash
python scripts/gcope/ete.py -c scripts/gcope/configs/ete_smoke.yaml
```

预期产物：

```text
storage/gcope/ete_smoke/cora_results.txt
storage/gcope/ete_smoke/results.pt
storage/gcope/ete_smoke/config.json
```

配置含义：

```text
stage: ete
target dataset: Cora
training mode: end-to-end supervised baseline
```

## PyGFM 统一入口

除直接运行 `scripts/gcope/*.py` 外，也可以通过 PyGFM 统一 CLI 调用：

```bash
pygfm -c scripts/gcope/configs/pretrain_smoke.yaml
pygfm -c scripts/gcope/configs/finetune_smoke.yaml
pygfm -c scripts/gcope/configs/prog_smoke.yaml
pygfm -c scripts/gcope/configs/ete_smoke.yaml
```
