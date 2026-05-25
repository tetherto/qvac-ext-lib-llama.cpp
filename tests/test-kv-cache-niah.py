#!/usr/bin/env python3
"""
NIAH (Needle In A Haystack) KV cache quality benchmark orchestrator.

Schedules niah-bench.sh jobs across multiple GPUs in parallel, collects
per-cell (ctx × depth) CSV results, and produces a per-cache-config
heatmap plus a cross-config summary.

NIAH's output is fundamentally different from the other 3 evals — there's
no per-task aggregation, just a 2-D grid of binary recall scores. We:

    1. Render one ctx × depth heatmap per (model, K, V) cell.
    2. Produce a one-line summary per cell: "<grid average>%, <hits>/<total>".
    3. Cross-cell comparison table: each row is a (K, V) config, columns
       are the ctx lengths, cells are the row-average across depths at
       that ctx.

Usage:
    python3 tests/test-kv-cache-niah.py -c smoke --gpus 0,1
    python3 tests/test-kv-cache-niah.py -c full  --gpus 0,1 --output-dir results-niah

Config presets:
    smoke -  3 ctx × 3 depths,  subset quants    (5-cell heatmap per quant)
    small -  4 ctx × 5 depths,  subset quants
    full  -  5 ctx × 7 depths,  all quants       (canonical 35-cell gkamradt grid)
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
NH_BENCH = SCRIPT_DIR / "niah-bench.sh"

BPW = {
    "f16":    16.0, "q8_0":   8.5, "q4_0":   4.5,
    "tbq3_0": 4.25, "tbq4_0": 5.25,
    "pq3_0":  3.25, "pq4_0":  4.25,
}


def bpw_label(k, v):
    return f"K:{BPW.get(k, '?')} V:{BPW.get(v, '?')}"


QUANT_CONFIGS_ALL = [
    ("f16", "f16"),
    ("tbq4_0", "pq4_0"),
    ("tbq3_0", "pq3_0"),
    ("tbq3_0", "q4_0"),
    ("tbq4_0", "q4_0"),
    ("pq4_0", "pq4_0"),
    ("pq3_0", "pq3_0"),
    ("q4_0", "q4_0"),
]
QUANT_CONFIGS_SUBSET = [
    ("f16", "f16"),
    ("tbq4_0", "pq4_0"),
    ("tbq3_0", "pq3_0"),
    ("tbq4_0", "q4_0"),
    ("q4_0", "q4_0"),
]


@dataclass
class ModelDef:
    path: str
    tokenizer: str
    label: str
    family: str


MODELS_DEFAULT = [
    ModelDef(
        path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        tokenizer="NousResearch/Meta-Llama-3.1-8B-Instruct",
        label="Llama-3.1-8B",
        family="Llama-3.1-8B",
    ),
    ModelDef(
        path="models/Ministral-8B-Instruct-2410-Q4_K_M.gguf",
        tokenizer="mistralai/Ministral-8B-Instruct-2410",
        label="Ministral-8B",
        family="Ministral-8B",
    ),
]


@dataclass
class ConfigPreset:
    name: str
    ctx_lengths: list[int]
    depth_percents: list[float]
    quant_configs: list = field(default_factory=list)


PRESETS = {
    # Small grid for plumbing. 5 cells per quant × 5 quants = 25 inferences total.
    "smoke": ConfigPreset("smoke",
                          [2048, 4096, 8192],
                          [25, 50, 75],
                          QUANT_CONFIGS_SUBSET),
    "small": ConfigPreset("small",
                          [2048, 4096, 8192, 16384],
                          [0, 25, 50, 75, 100],
                          QUANT_CONFIGS_SUBSET),
    # The canonical gkamradt-style grid (5 ctx × 7 depths). 32k context
    # needs ~17 GB of f16 KV; comfortable on a 32 GB 5090 but the largest
    # cell takes ~1-2 minutes by itself.
    "full":  ConfigPreset("full",
                          [2048, 4096, 8192, 16384, 32768],
                          [0, 10, 25, 50, 75, 90, 100],
                          QUANT_CONFIGS_ALL),
}


@dataclass
class Job:
    model: ModelDef
    k_type: str
    v_type: str
    ctx_lengths: list[int]
    depth_percents: list[float]
    gpu_id: int = 0
    csv_file: str = ""


_print_lock = threading.Lock()


class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def tick(self):
        with self.lock:
            self.completed += 1
            elapsed = time.time() - self.start_time
            avg = elapsed / self.completed if self.completed else 0
            rem = max(0, self.total - self.completed)
            eta = _fmt_duration(avg * rem) if self.completed else "?"
            return f"[{self.completed}/{self.total}] elapsed {_fmt_duration(elapsed)}, ETA {eta}"


def _fmt_duration(secs):
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


_progress: Optional[ProgressTracker] = None


def run_job(job: Job, extra_args, output_dir: Path):
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_name = f"{job.model.label}_{job.k_type}_{job.v_type}".replace("/", "_").replace(" ", "_")
    csv_path = str(jobs_dir / f"{job_name}.csv")
    job.csv_file = csv_path

    env = os.environ.copy()
    env["GGML_VK_VISIBLE_DEVICES"] = str(job.gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)

    cmd = [
        "bash", str(NH_BENCH),
        "-m", job.model.path,
        "-ctk", job.k_type,
        "-ctv", job.v_type,
        "--ctx-lengths", " ".join(str(c) for c in job.ctx_lengths),
        "--depth-percents", " ".join(str(d) for d in job.depth_percents),
        "--tokenizer", job.model.tokenizer,
        "--csv", csv_path,
    ]
    cmd.extend(extra_args)

    label = f"[GPU{job.gpu_id}] {job.model.label} K={job.k_type} V={job.v_type}"
    log_path = csv_path.replace(".csv", ".txt")
    with _print_lock:
        logger.info("  START: %s", label)
        logger.info("  $ %s", " ".join(cmd))
        logger.info("  log: %s", log_path)
        logger.info("")

    try:
        # `errors="replace"` — see longbench orchestrator for rationale.
        result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                errors="replace", timeout=6 * 3600)
        if result.returncode != 0:
            with _print_lock:
                logger.info("  FAIL:  %s", label)
                logger.info("         stderr: %s", result.stderr[-500:])
            return None
    except subprocess.TimeoutExpired:
        with _print_lock:
            logger.info("  TIMEOUT: %s", label)
        return None

    grid_avg = "?"
    for line in result.stdout.splitlines():
        if "Grid average:" in line:
            grid_avg = line.strip().split("Grid average:")[1].strip()
            break

    progress_str = _progress.tick() if _progress else ""
    with _print_lock:
        logger.info("  DONE:  %s  =>  %s  %s", label, grid_avg, progress_str)
    return csv_path


@dataclass
class CellResult:
    model: str
    cache_k: str
    cache_v: str
    ctx_len: int
    depth_percent: float
    score: float  # percent


def collect_results(csv_paths):
    out = []
    for path in csv_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    out.append(CellResult(
                        model=row["model"],
                        cache_k=row["cache_k"], cache_v=row["cache_v"],
                        ctx_len=int(row["ctx_len"]),
                        depth_percent=float(row["depth_percent"]),
                        score=float(row["score"]),
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning("  WARN: bad CSV row in %s: %s", path, e)
    return out


def heatmap_for_cell(results, model, k, v, ctx_lengths, depth_percents):
    """Build a 2-D grid of scores for one (model, K, V) cell."""
    lookup = {}
    for r in results:
        if r.model == model and r.cache_k == k and r.cache_v == v:
            lookup[(r.ctx_len, r.depth_percent)] = r.score
    grid = []
    for c in ctx_lengths:
        row = []
        for d in depth_percents:
            row.append(lookup.get((c, d)))
        grid.append(row)
    return grid


def render_heatmap(results, ctx_lengths, depth_percents, title):
    if not results:
        return f"\n{title}\n  (no results)\n"
    models = sorted({r.model for r in results})
    quants = sorted({(r.cache_k, r.cache_v) for r in results},
                    key=lambda kv: (BPW.get(kv[0], 99) + BPW.get(kv[1], 99)))
    lines = ["", "=" * 100, f" {title}", "=" * 100]
    for model_name in models:
        for k, v in quants:
            lines.append("")
            lines.append(f"  {model_name}  K={k} V={v}  ({bpw_label(k, v)})")
            header = f"    {'ctx ↓':<10}"
            for d in depth_percents:
                header += f"{int(d)}%".rjust(7)
            header += "  row-avg"
            lines.append(header)
            grid = heatmap_for_cell(results, model_name, k, v, ctx_lengths, depth_percents)
            row_avgs = []
            for ci, c in enumerate(ctx_lengths):
                row = f"    {c:<10}"
                vals = []
                for di, d in enumerate(depth_percents):
                    cell = grid[ci][di]
                    if cell is None:
                        row += "    -- "
                    else:
                        row += f"{int(cell):4d}% "
                        vals.append(cell)
                if vals:
                    avg = sum(vals) / len(vals)
                    row += f"  {avg:5.1f}%"
                    row_avgs.append(avg)
                else:
                    row += "    -- "
                lines.append(row)
            if row_avgs:
                lines.append(f"    grid-avg: {sum(row_avgs)/len(row_avgs):5.1f}%")
    lines.append("=" * 100)
    return "\n".join(lines)


def render_comparison(results, ctx_lengths, title):
    """One row per (model, K, V), columns are ctx lengths (row-mean across depths)."""
    if not results:
        return ""
    models = sorted({r.model for r in results})
    quants = sorted({(r.cache_k, r.cache_v) for r in results},
                    key=lambda kv: (BPW.get(kv[0], 99) + BPW.get(kv[1], 99)))
    lines = ["", "=" * 110, f" {title}", "=" * 110]
    for model_name in models:
        lines.append("")
        lines.append(f"  Model: {model_name}")
        header = f"  {'KV Config':<22}{'BPW':>6}"
        for c in ctx_lengths:
            header += f"{c:>10}"
        header += f"{'Grid avg':>12}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for k, v in quants:
            avg_bpw = (BPW.get(k, 0) + BPW.get(v, 0)) / 2
            row = f"  {k}/{v:<14}{avg_bpw:>6.2f}"
            ctx_means = []
            for c in ctx_lengths:
                hits = [r.score for r in results
                        if r.model == model_name and r.cache_k == k
                        and r.cache_v == v and r.ctx_len == c]
                if hits:
                    m = sum(hits) / len(hits)
                    row += f"{m:>10.1f}"
                    ctx_means.append(m)
                else:
                    row += f"{'--':>10}"
            if ctx_means:
                grid = sum(ctx_means) / len(ctx_means)
                row += f"{grid:>12.1f}"
            else:
                row += f"{'--':>12}"
            lines.append(row)
    lines.append("=" * 110)
    return "\n".join(lines)


def write_combined_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "cache_k", "cache_v", "k_bpw", "v_bpw",
                    "ctx_len", "depth_percent", "score_pct"])
        for r in results:
            w.writerow([r.model, r.cache_k, r.cache_v,
                        BPW.get(r.cache_k, "?"), BPW.get(r.cache_v, "?"),
                        r.ctx_len, r.depth_percent, r.score])


def _next_available_dir(base):
    if not base.exists() or not any(base.iterdir()):
        return base
    i = 1
    while True:
        c = base.parent / f"{base.name}{i}"
        if not c.exists() or not any(c.iterdir()):
            return c
        i += 1


def _find_latest_dir(base):
    latest = None
    if base.exists() and any(base.iterdir()):
        latest = base
    i = 1
    while True:
        c = base.parent / f"{base.name}{i}"
        if c.exists() and any(c.iterdir()):
            latest = c
            i += 1
        else:
            break
    return latest


def _job_csv_path(output_dir, job):
    name = f"{job.model.label}_{job.k_type}_{job.v_type}".replace("/", "_").replace(" ", "_")
    return output_dir / "jobs" / f"{name}.csv"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="NIAH KV cache quality orchestrator")
    parser.add_argument("-c", "--config", default="full", choices=PRESETS.keys())
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--output-dir", default="results-niah")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-missing", action="store_true")
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    preset = PRESETS[args.config]
    gpu_ids = [int(g) for g in args.gpus.split(",")]

    base_dir = Path(args.output_dir)
    if args.rerun_missing:
        output_dir = _find_latest_dir(base_dir) or base_dir
        if output_dir == base_dir and not base_dir.exists():
            logger.info("No existing '%s' dir; creating fresh.", base_dir)
        else:
            logger.info("Rerun-missing: reusing %s", output_dir)
    else:
        output_dir = _next_available_dir(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        models = []
        for p in args.models:
            for m in MODELS_DEFAULT:
                if m.path == p:
                    models.append(m)
                    break
            else:
                models.append(ModelDef(
                    path=p,
                    tokenizer="NousResearch/Meta-Llama-3.1-8B-Instruct",
                    label=Path(p).stem,
                    family="custom",
                ))
    else:
        models = MODELS_DEFAULT

    available = [m for m in models if os.path.exists(m.path)]
    if not available:
        logger.error("ERROR: No model files found:")
        for m in models:
            logger.error("  %s", m.path)
        sys.exit(1)
    missing = [m for m in models if not os.path.exists(m.path)]
    if missing:
        logger.warning("WARNING: Skipping missing models:")
        for m in missing:
            logger.warning("  %s", m.path)

    all_jobs = []
    extra_args = ["-ngl", "99", "-fa", "1", "--split-mode", "none"] + args.extra
    for model in available:
        for k, v in preset.quant_configs:
            all_jobs.append(Job(model=model, k_type=k, v_type=v,
                                ctx_lengths=preset.ctx_lengths,
                                depth_percents=preset.depth_percents))

    skipped = []
    if args.rerun_missing:
        remaining = []
        for j in all_jobs:
            p = _job_csv_path(output_dir, j)
            if p.exists() and p.stat().st_size > 50:
                logger.info("  SKIP (exists): %s K=%s V=%s",
                            j.model.label, j.k_type, j.v_type)
                skipped.append(str(p))
            else:
                remaining.append(j)
        all_jobs = remaining
    for i, j in enumerate(all_jobs):
        j.gpu_id = gpu_ids[i % len(gpu_ids)]

    logger.info("")
    logger.info("  $ python3 %s", " ".join(sys.argv))
    logger.info("")
    logger.info("=" * 60)
    logger.info(" KV Cache NIAH Benchmark")
    logger.info("=" * 60)
    logger.info("  Config:      %s", preset.name)
    logger.info("  ctx_lengths: %s", preset.ctx_lengths)
    logger.info("  depths (%%): %s", preset.depth_percents)
    logger.info("  GPUs:        %s", gpu_ids)
    logger.info("  Models:      %d", len(available))
    for m in available:
        logger.info("    - %s: %s", m.label, m.path)
    logger.info("  Quants:      %d", len(preset.quant_configs))
    for k, v in preset.quant_configs:
        logger.info("    - K=%s V=%s (%s)", k, v, bpw_label(k, v))
    logger.info("  Jobs:        %d to run%s",
                len(all_jobs),
                f" ({len(skipped)} skipped)" if skipped else "")
    logger.info("  Output:      %s", output_dir)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("\nDRY RUN — jobs that would be scheduled:\n")
        for j in all_jobs:
            logger.info("  [GPU%d] %s K=%s V=%s ctx=%s depths=%s",
                        j.gpu_id, j.model.label, j.k_type, j.v_type,
                        j.ctx_lengths, j.depth_percents)
        return

    csv_paths = list(skipped)
    if all_jobs:
        global _progress
        _progress = ProgressTracker(len(all_jobs))
        logger.info("\nRunning %d jobs across %d GPU(s)...\n", len(all_jobs), len(gpu_ids))
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = {pool.submit(run_job, j, extra_args, output_dir): j for j in all_jobs}
            for fut in as_completed(futures):
                p = fut.result()
                if p:
                    csv_paths.append(p)

    results = collect_results(csv_paths)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not results:
        logger.error("No results collected. Inspect %s/jobs/*.txt for failures.", output_dir)
        sys.exit(1)

    tables = []
    t1 = render_comparison(results, preset.ctx_lengths,
                           "NIAH — cross-config summary (row-avg across depths per ctx)")
    logger.info(t1)
    tables.append(t1)
    t2 = render_heatmap(results, preset.ctx_lengths, preset.depth_percents,
                        "NIAH — per-cell ctx × depth heatmap")
    logger.info(t2)
    tables.append(t2)

    combined = output_dir / f"kv-niah_{preset.name}_{timestamp}.csv"
    write_combined_csv(results, str(combined))
    logger.info("\nCombined CSV: %s", combined)

    report = output_dir / f"kv-niah_report_{preset.name}_{timestamp}.txt"
    with open(report, "w") as f:
        f.write("KV Cache NIAH Benchmark Report\n")
        f.write(f"Config: {preset.name}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Command: python3 {' '.join(sys.argv)}\n")
        f.write(f"Models: {', '.join(m.label for m in available)}\n\n")
        for t in tables:
            f.write(t)
            f.write("\n\n")
    logger.info("Report: %s", report)
    logger.info("Job logs/CSVs: %s/", output_dir / "jobs")
    logger.info("Done.")


if __name__ == "__main__":
    main()
