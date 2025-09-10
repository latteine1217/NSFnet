"""
Legacy checkpoint converter

This script converts paired legacy checkpoints (.pth and .pth_evm) into the
unified checkpoint format expected by the current PINN solver (`load_checkpoint`).

Usage:
  python pinn_modules/convert_legacy_checkpoints.py \
    --input_dir results/Re5000/4x120_Nf200k_lamB10_alpha0.05 \
    --output_dir results/converted/Re5000/4x120_Nf200k_lamB10_alpha0.05 \
    --re 5000 --alpha_evm 0.05

Notes:
- Epoch is inferred from filename patterns such as `loop(\d+)` or `epoch_(\d+)`.
- If `--re` or `--alpha_evm` are not provided, the script tries to parse them
  from the input path (e.g., `Re5000`, `alpha0.05`).
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Optional, Tuple, Dict, Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert legacy .pth/.pth_evm to unified checkpoints")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing legacy .pth and .pth_evm files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write unified checkpoints")
    parser.add_argument("--re", type=int, default=None, help="Reynolds number; if omitted, try parse from path (e.g., Re5000)")
    parser.add_argument("--alpha_evm", type=float, default=None, help="alpha_evm; if omitted, try parse from path (e.g., alpha0.03)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing unified checkpoints if present")
    parser.add_argument("--dry_run", action="store_true", help="List planned conversions without writing files")
    return parser.parse_args()


def _infer_epoch_from_name(name: str) -> Optional[int]:
    """Infer epoch from filename or path.

    Supports patterns: `loop(\d+)`, `epoch_(\d+)`.
    """
    for pat in (r"loop(\d+)", r"epoch_(\d+)"):
        m = re.search(pat, name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None


def _infer_re_from_path(path: str) -> Optional[int]:
    m = re.search(r"Re(\d+)", path)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _infer_alpha_from_path(path: str) -> Optional[float]:
    # match alpha0.05 or alpha_0.05
    m = re.search(r"alpha[_]?([0-9]+(?:\.[0-9]+)?)", path)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _find_evm_pair(main_path: str) -> Optional[str]:
    """Find the matching .pth_evm file for a given main .pth.

    Tries `<file>.pth_evm` and replacing suffix to `.pth_evm`.
    """
    if main_path.endswith(".pth"):
        candidate = main_path + "_evm"
        if os.path.isfile(candidate):
            return candidate
        alt = main_path[:-4] + ".pth_evm"
        if os.path.isfile(alt):
            return alt
    return None


def _is_unified_checkpoint(path: str) -> bool:
    return bool(re.match(r"checkpoint_epoch_\d+\.pth$", os.path.basename(path)))


def convert_directory(
    input_dir: str,
    output_dir: str,
    default_re: Optional[int] = None,
    default_alpha: Optional[float] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Convert all legacy pairs inside input_dir into unified checkpoints.

    Returns a summary dict with counts and skipped reasons.
    """
    os.makedirs(output_dir, exist_ok=True)

    inferred_re = default_re if default_re is not None else _infer_re_from_path(input_dir)
    inferred_alpha = default_alpha if default_alpha is not None else _infer_alpha_from_path(input_dir)

    summary = {
        "pairs": 0,
        "written": 0,
        "skipped_no_pair": 0,
        "skipped_exists": 0,
        "errors": 0,
    }

    entries = sorted(os.listdir(input_dir))
    for fname in entries:
        # only consider main .pth files; skip evm and already unified
        if not fname.endswith(".pth"):
            continue
        if fname.endswith(".pth_evm"):
            continue
        if _is_unified_checkpoint(fname):
            continue

        main_path = os.path.join(input_dir, fname)
        evm_path = _find_evm_pair(main_path)
        if not evm_path:
            summary["skipped_no_pair"] += 1
            print(f"[skip] No matching EVM file for: {fname}")
            continue

        epoch = _infer_epoch_from_name(fname) or _infer_epoch_from_name(main_path) or 0
        out_name = f"checkpoint_epoch_{epoch}.pth"
        out_path = os.path.join(output_dir, out_name)

        if os.path.exists(out_path) and not overwrite:
            summary["skipped_exists"] += 1
            print(f"[skip] Exists: {out_name} (use --overwrite to replace)")
            continue

        summary["pairs"] += 1

        print("=== Converting ===")
        print(f"main: {main_path}")
        print(f" evm: {evm_path}")
        print(f" ->  {out_path}")

        if dry_run:
            continue

        try:
            main_state = torch.load(main_path, map_location="cpu")
            evm_state = torch.load(evm_path, map_location="cpu")

            # unified checkpoint payload
            checkpoint = {
                "epoch": epoch,
                "net_state_dict": main_state,
                "net_1_state_dict": evm_state,
                "optimizer_state_dict": {},
                "Re": inferred_re,
                "alpha_evm": inferred_alpha,
                "current_stage": " _converted",  # keep a space for compatibility
                "global_step_offset": 0,
                "current_weight_decay": 0.0,
            }
            torch.save(checkpoint, out_path)
            summary["written"] += 1
        except Exception as e:
            summary["errors"] += 1
            print(f"[error] Failed to convert {fname}: {e}")

    print("=== Summary ===")
    print(f"pairs:   {summary['pairs']}")
    print(f"written: {summary['written']}")
    print(f"exists:  {summary['skipped_exists']}")
    print(f"no_pair: {summary['skipped_no_pair']}")
    print(f"errors:  {summary['errors']}")
    return summary


def main() -> None:
    args = parse_args()
    convert_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        default_re=args.re,
        default_alpha=args.alpha_evm,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

