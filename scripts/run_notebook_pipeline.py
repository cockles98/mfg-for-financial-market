#!/usr/bin/env python
"""
Execute o pipeline completo sem depender do Jupyter Notebook.

O script reproduz `notebooks/mfg_pipeline.ipynb`: lê a configuração baseline,
executa o solver com clearing endógeno e grava todos os artefatos em
`notebooks_output/run-YYYYmmdd-HHMMSS/`.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from mfg_finance.cli import _run_single_experiment


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute o pipeline completo e salve os artefatos no diretório de notebooks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Arquivo de configuração YAML usado pelo solver.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("notebooks_output"),
        help="Diretório raiz onde o run será armazenado.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=datetime.now().strftime("run-%Y%m%d-%H%M%S"),
        help="Nome do subdiretório destino (timestamp).",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_root / args.timestamp
    if output_dir.exists():
        shutil.rmtree(output_dir)

    stats = _run_single_experiment(cfg, output_dir, compute_price=True)
    price_mean = stats.get("price_mean")
    price_std = stats.get("price_std")

    print(f"Notebook artifacts stored in {output_dir}")
    print(
        f"Iterations: {stats['iterations']}, "
        f"final_error: {stats['final_error']:.3e}, "
        f"price_mean: {price_mean}, price_std: {price_std}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
