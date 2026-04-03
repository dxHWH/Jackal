# DRQN Artifact Layout

This project now saves DRQN artifacts per run to avoid model/config mismatches.

## New layout

- `DRQN_Test/models/<run_name>/run_config.json`
- `DRQN_Test/models/<run_name>/model_final.pth`
- `DRQN_Test/models/<run_name>/checkpoints/ep<episode>.pth`

Example:

- `DRQN_Test/models/drqn_manualaim_1v1_run1/run_config.json`
- `DRQN_Test/models/drqn_manualaim_1v1_run1/model_final.pth`
- `DRQN_Test/models/drqn_manualaim_1v1_run1/checkpoints/ep100.pth`

## Why

The previous flat layout put all models under one folder and reused one shared config file. That could pair a model with the wrong config.

## Train

```bash
python DRQN_train.py --save-prefix drqn_manualaim_1v1_run1
```

## Test

```bash
python DRQN_test_render.py --model-path DRQN_Test/models/drqn_manualaim_1v1_run1/model_final.pth --episodes 3
```

## Backward compatibility

`DRQN_test_render.py` still supports legacy flat files and tries to resolve configs in this order:

1. same directory: `run_config.json`
2. legacy sibling config inferred from model name
3. fallback: `drqn_run_config.json`
