# Colab Phase 0 Runbook

Use this after uploading the project folder to Colab.

## 1) Open runtime and go to project root

```bash
cd /content/midmamba
python --version
```

## 2) Install Phase 0 dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-colab.txt
```

## 3) Verify GPU visibility

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
```

Expected for your target setup: capability should print `(12, 0)`.

## 4) Run all Phase 0 checks in one command

```bash
python scripts/phase0_colab_checks.py
```

This writes a machine-readable report to:

- `results/phase0_colab_check_results.json`

## 5) If DBN check is blocked

The DBN inspection check needs at least one local `.dbn.zst` file under `data/`.

- If no DBN files are present, keep data checks as blocked and continue with env checks.
- Once one DBN file is downloaded into `data/`, rerun:

```bash
python scripts/phase0_colab_checks.py
```

## 6) Update tracker file

Copy key pass/fail outcomes into:

- `environment_check.md`

Include:

- GPU name and compute capability
- Mamba forward/backward pass result
- torchao FP8 conversion result
- DBN inspection output status
