#!/usr/bin/env python3
"""Run Colab/GPU sanity checks for the RL execution project."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def first_dataframe(value: object):
    import pandas as pd  # type: ignore

    if isinstance(value, pd.DataFrame):
        return value
    try:
        first = next(iter(value))  # type: ignore[arg-type]
    except StopIteration as exc:
        raise ValueError("DBNStore.to_df returned no DataFrame chunks") from exc
    if not isinstance(first, pd.DataFrame):
        raise TypeError(f"DBNStore.to_df returned {type(first).__name__}, expected DataFrame")
    return first


def run_torch_gpu_check() -> dict:
    result: dict = {"name": "torch_gpu", "ok": False}
    try:
        import torch  # type: ignore

        result["torch_version"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["device_name"] = torch.cuda.get_device_name(0)
            result["capability"] = torch.cuda.get_device_capability(0)
        result["ok"] = True
    except Exception as exc:  # pragma: no cover
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_mamba_check() -> dict:
    result: dict = {"name": "mamba_forward_backward", "ok": False}
    try:
        import torch  # type: ignore
        from mamba_ssm import Mamba2  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Mamba2(d_model=128).to(device)
        x = torch.randn(2, 512, 128, device=device, requires_grad=True)
        y = model(x)
        loss = y.square().mean()
        loss.backward()
        result["device"] = device
        result["output_shape"] = list(y.shape)
        result["loss"] = float(loss.detach().cpu().item())
        result["ok"] = True
    except Exception as exc:  # pragma: no cover
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=3)
    return result


def run_torchao_fp8_check() -> dict:
    result: dict = {"name": "torchao_fp8", "ok": False}
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training  # type: ignore

        if not torch.cuda.is_available():
            result["error"] = "CUDA unavailable; FP8 check requires GPU runtime."
            return result

        device = "cuda"
        model = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 512)).to(device).bfloat16()

        def fp8_filter(mod: nn.Module, fqn: str) -> bool:
            return (
                isinstance(mod, nn.Linear)
                and mod.in_features % 16 == 0
                and mod.out_features % 16 == 0
            )

        fp8_config = Float8LinearConfig.from_recipe_name("rowwise")
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_filter)
        model = torch.compile(model)

        # Keep dimensions FP8-safe for scaled_mm (multiples of 16).
        x = torch.randn(16, 512, device=device, dtype=torch.bfloat16, requires_grad=True)
        y = model(x)
        loss = y.square().mean()
        loss.backward()
        result["loss"] = float(loss.detach().cpu().item())
        result["ok"] = True
    except Exception as exc:  # pragma: no cover
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=3)
    return result


def run_dbn_inspect_check(root: Path) -> dict:
    result: dict = {"name": "dbn_inspect", "ok": False}
    try:
        import databento as db  # type: ignore
    except Exception as exc:  # pragma: no cover
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    candidates = sorted(root.glob("data/**/*.dbn.zst"))
    if not candidates:
        result["error"] = "No .dbn.zst files found under data/."
        return result

    target = candidates[0]
    try:
        store = db.DBNStore.from_file(str(target))
        df = first_dataframe(store.to_df(count=5))
        result["file"] = str(target)
        result["sample_row_count"] = int(len(df))
        result["column_count"] = int(len(df.columns))
        result["columns"] = [str(c) for c in list(df.columns)]
        result["index_type"] = type(df.index).__name__
        result["first_index"] = str(df.index[0]) if len(df.index) > 0 else None
        if "instrument_id" in df.columns:
            result["instrument_id_sample"] = [int(v) for v in df["instrument_id"].head(5).tolist()]
        result["ok"] = True
    except Exception as exc:  # pragma: no cover
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Colab/GPU environment checks.")
    parser.add_argument(
        "--output-json",
        default="results/env_check_results.json",
        help="Path for JSON results output (default: results/env_check_results.json)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_path = (root / args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checks = [
        run_torch_gpu_check(),
        run_mamba_check(),
        run_torchao_fp8_check(),
        run_dbn_inspect_check(root),
    ]

    for check in checks:
        print_section(check["name"])
        print(json.dumps(check, indent=2))

    output_path.write_text(json.dumps({"checks": checks}, indent=2))
    print(f"\nSaved results JSON: {output_path}")

    # Return non-zero if any check fails; useful in CI/automation.
    return 0 if all(bool(c.get("ok")) for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
