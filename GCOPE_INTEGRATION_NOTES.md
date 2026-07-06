# GCOPE 集成到 PyGFM 的注意事项

## 当前集成方式

本项目采用低侵入方式将 GCOPE 集成到 PyGFM：

- GCOPE 原始代码迁入 `src/pygfm/baseline_models/gcope/original_src/`
- 新增 `src/pygfm/baseline_models/gcope/runner.py`
- 新增 `src/pygfm/cli/baselines/gcope.py`
- 新增 `scripts/gcope/` 及 YAML 配置

`runner.py` 会把 PyGFM 风格 YAML 转换为 GCOPE 原始 `fastargs` 参数，例如：

```text
general.func -> --general.func
data.name -> --data.name
pretrain.cross_link -> --pretrain.cross_link
```

## 支持入口

直接脚本入口：

```bash
python scripts/gcope/pretrain.py -c scripts/gcope/configs/pretrain_smoke.yaml
python scripts/gcope/finetune.py -c scripts/gcope/configs/finetune_smoke.yaml
python scripts/gcope/prog.py -c scripts/gcope/configs/prog_smoke.yaml
python scripts/gcope/ete.py -c scripts/gcope/configs/ete_smoke.yaml
```

PyGFM 统一入口：

```bash
pygfm -c scripts/gcope/configs/pretrain_smoke.yaml
pygfm -c scripts/gcope/configs/ete_smoke.yaml
```

## 主要风险

1. GCOPE 原代码依赖 `fastargs` 全局状态，当前适配方式适合单进程单次运行一个实验。
2. GCOPE 原代码大量使用 `from data import ...`、`from model import ...` 等短导入，当前通过把 `original_src` 加入 `sys.path` 兼容。
3. 预训练 checkpoint 文件名由源数据集顺序决定，例如 `cora,citeseer_pretrained_model.pt`。
4. 原 `adapt.py` 中存在固定 `cuda(1)` 的加载方式，单 GPU 或 CPU 环境建议改为 `cuda:0` 或 `cpu`。
5. 原 `pretrain.json` 不是严格 JSON，建议使用本项目提供的 YAML 配置。
6. `BWGNN` 依赖 DGL、NetworkX、SciPy、SymPy，建议先用 GCN/GAT/FAGCN 验证流程。

## 后续重构建议

第一步：保持当前 wrapper 方式，完成 smoke test 和代表性论文实验复现。

第二步：把 `original_src/data`、`model`、`functional` 改成真正的包内相对导入。

第三步：去掉 `fastargs`，训练函数改为显式接收 `cfg`。

第四步：让 GCOPE 模型继承 PyGFM 的 `GFMModelBase`，并复用 PyGFM 公共 loss、data utility。
