# GCOPE2PyGFM 提交说明

## 项目简介

本项目在 PyGFM 基础上集成 GCOPE，使 GCOPE 可以作为 PyGFM 的一个 Graph Foundation Model baseline 使用。当前集成保留 GCOPE 原始算法代码，并新增 PyGFM 风格 YAML 配置、脚本入口和 CLI 注册。

## 主要内容

- `src/pygfm/baseline_models/gcope/`
  - `original_src/`：迁入的 GCOPE 原始源码。
  - `runner.py`：PyGFM YAML 到 GCOPE fastargs 参数的适配层。
  - `__init__.py`：GCOPE baseline 包入口。

- `src/pygfm/cli/baselines/gcope.py`
  - 在 PyGFM registry 中注册 `gcope/pretrain`、`gcope/finetune`、`gcope/prog`、`gcope/ete`。

- `scripts/gcope/`
  - `pretrain.py`：GCOPE 预训练入口。
  - `finetune.py`：GCOPE finetune 迁移入口。
  - `prog.py`：GCOPE ProG prompt 迁移入口。
  - `ete.py`：端到端监督训练入口。
  - `configs/*.yaml`：对应 smoke test 配置。

- `GCOPE_INTEGRATION_NOTES.md`
  - 说明 GCOPE 集成到 PyGFM 的注意事项和后续重构建议。

## 运行方式

安装依赖：

```bash
pip install -e ".[torch,gcope]"
```

直接运行：

```bash
python scripts/gcope/pretrain.py -c scripts/gcope/configs/pretrain_smoke.yaml
python scripts/gcope/finetune.py -c scripts/gcope/configs/finetune_smoke.yaml
python scripts/gcope/prog.py -c scripts/gcope/configs/prog_smoke.yaml
python scripts/gcope/ete.py -c scripts/gcope/configs/ete_smoke.yaml
```

通过 PyGFM 统一入口运行：

```bash
pygfm -c scripts/gcope/configs/pretrain_smoke.yaml
pygfm -c scripts/gcope/configs/ete_smoke.yaml
```

## 数据集说明

提交包不包含数据集。GCOPE 使用 PyTorch Geometric 数据集接口自动下载或读取缓存，缓存目录由 YAML 中的 `general.cache_dir` 控制。

## 验证内容

已完成静态验证：

- 新增 Python 文件可通过编译检查。
- PyGFM registry 可识别 `gcope/pretrain`、`gcope/finetune`、`gcope/prog` 和 `gcope/ete`。
- YAML 配置可以转换为 GCOPE 原始 `fastargs` 参数格式。
