# AGENTS.md

This repository is a research codebase built on two upstream projects:

- `mmdetection3d/` (OpenMMLab + MMEngine; lint + pytest suite present)
- `FSHNet/` (OpenPCDet-style; training/eval scripts; no repo-wide lint/test harness)

Work inside the subtree you are modifying and follow that subtree's conventions.

Cursor rules: none found (`.cursor/rules/`, `.cursorrules`).
Copilot instructions: none found (`.github/copilot-instructions.md`).

## Commands

### mmdetection3d

```bash
# from mmdetection3d/

# build / install
pip install -e .

# lint / format (preferred)
pre-commit run --all-files

# lint / format (manual; matches mmdetection3d/.dev_scripts/linter.sh)
yapf -r -i mmdet3d/ configs/ tests/ tools/
isort mmdet3d/ configs/ tests/ tools/
flake8 .

# tests
pytest tests/

# single test file
pytest tests/test_models/test_voxel_encoders/test_pillar_encoder.py

# single test function
pytest tests/test_models/test_voxel_encoders/test_pillar_encoder.py::test_pillar_encoder

# filter by keyword
pytest -k "pillar and not cuda" tests/

# coverage (matches CI)
coverage run --branch --source mmdet3d -m pytest tests/
coverage report -m

# train / eval
python tools/train.py <config.py>
python tools/test.py <config.py> <checkpoint.pth>

# multi-GPU
bash tools/dist_train.sh <config.py> <num_gpus>
bash tools/dist_test.sh <config.py> <checkpoint.pth> <num_gpus>
```

### FSHNet

```bash
# from FSHNet/

# build / install
pip install -r requirements.txt
python setup.py develop

# train / eval
python tools/train.py --cfg_file tools/cfgs/<...>.yaml
python tools/test.py --cfg_file tools/cfgs/<...>.yaml --ckpt <path>

# multi-GPU
bash tools/scripts/dist_train.sh <num_gpus> --cfg_file tools/cfgs/<...>.yaml
```

## Code Style

### mmdetection3d (OpenMMLab conventions)

Primary sources of truth:

- `mmdetection3d/setup.cfg` (yapf/isort/flake8/codespell)
- `mmdetection3d/.pre-commit-config.yaml` (pre-commit hooks)

Formatting:

- Use `yapf` (pep8-based); do not hand-format.
- Prefer <=79 chars where practical (isort/docformatter wrapping).
- Let `pre-commit` fix whitespace and docs formatting.

Imports:

- Use `isort`; import groups are stdlib, third-party, first-party.
- First-party import root is `mmdet3d`.

Linting:

- Use `flake8`.
- Some config files intentionally ignore F401/F403/F405 (see `setup.cfg`).

Typing:

- Add/maintain type hints (heavily used throughout `mmdet3d/`).
- Prefer precise types (`Optional[...]`, `Dict[...]`, `Sequence[...]`) over `Any`.

Naming:

- PEP 8: `snake_case` for funcs/vars, `PascalCase` for classes.
- Keep names consistent with the config/registry naming.

Architecture patterns:

- Registry-based construction is standard:
  `@MODELS.register_module()`, `@DATASETS.register_module()`, etc.
- Wire new components through the registry + config system (avoid ad-hoc globals).

Error handling:

- Raise `ValueError`/`TypeError`/`KeyError` with actionable messages.
- `assert` is fine for internal invariants (common pattern here).
- Avoid bare `except:` and do not swallow exceptions.

Logging:

- Prefer MMEngine logging (`mmengine.logging.MMLogger`, `print_log`) for user-facing
  messages and training logs.
- If a module already uses `print()` (common in evaluation utilities), keep changes
  minimal; do not refactor logging unless required by the task.

### FSHNet (OpenPCDet-style conventions)

- Keep diffs small and localized; avoid large-scale reformatting.
- Relative imports and script-like modules are common; follow the local file style.
- Type hints are not consistently enforced; only add when clearly beneficial.

## Agent Operating Rules

- Decide which subtree you are editing first (`mmdetection3d/` vs `FSHNet/`).
- Bugfix rule: fix minimally; do not refactor while fixing.
- Before finishing: run the relevant commands for the subtree you touched.
