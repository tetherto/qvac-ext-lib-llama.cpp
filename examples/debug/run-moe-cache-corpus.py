#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


TELEMETRY_RE = re.compile(
    r"moe selected bytes = (?P<selected>\d+).*?"
    r"cache = (?P<hits>\d+)/(?P<misses>\d+) hits/misses, "
    r"(?P<evictions>\d+) evictions, (?P<fill>\d+) fill bytes"
)
PERF_RE = re.compile(
    r"eval time =\s+(?P<ms>[0-9.]+) ms /\s+(?P<runs>\d+) runs.*?"
    r"(?P<tps>[0-9.]+) tokens per second"
)
ID_RE = re.compile(r"^[a-z0-9-]+$")


def load_corpus(path):
    cases = []
    seen = set()
    with path.open(encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            for field in ("id", "regime", "language", "prompt"):
                if not isinstance(case.get(field), str) or not case[field]:
                    raise ValueError(f"{path}:{line_number}: invalid {field}")
            if not ID_RE.fullmatch(case["id"]):
                raise ValueError(f"{path}:{line_number}: invalid id {case['id']!r}")
            if case["id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {case['id']!r}")
            seen.add(case["id"])
            cases.append(case)
    return cases


def parse_metrics(stderr):
    telemetry = [match.groupdict() for match in TELEMETRY_RE.finditer(stderr)]
    perf = list(PERF_RE.finditer(stderr))
    result = {
        "telemetry_records": len(telemetry),
        "selected_bytes": sum(int(item["selected"]) for item in telemetry),
        "cache_hits": sum(int(item["hits"]) for item in telemetry),
        "cache_misses": sum(int(item["misses"]) for item in telemetry),
        "cache_evictions": sum(int(item["evictions"]) for item in telemetry),
        "fill_bytes": sum(int(item["fill"]) for item in telemetry),
    }
    if perf:
        result["eval_ms"] = float(perf[-1].group("ms"))
        result["eval_runs"] = int(perf[-1].group("runs"))
        result["tokens_per_second"] = float(perf[-1].group("tps"))
    return result


def run_case(args, case, mode, extra_args, output_dir):
    command = [
        str(args.binary),
        "--model", str(args.model),
        "--prompt", case["prompt"],
        "--n-predict", str(args.n_predict),
        "--no-warmup",
        "--tensor-filter", "^$",
        "--verbose",
    ]
    command.extend(extra_args)
    if mode == "cache":
        command.extend(["--moe-cache-mib", str(args.moe_cache_mib)])

    env = os.environ.copy()
    env["GGML_MOE_TRACE"] = "1"
    env.setdefault("GGML_OP_OFFLOAD_MIN_BATCH", "1")
    env.pop("GGML_MOE_COMPACT", None)
    env.pop("GGML_DEBUG_FULL", None)
    env.pop("LLAMA_ARG_MOE_CACHE_MIB", None)

    started = time.monotonic()
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    duration = time.monotonic() - started

    prefix = output_dir / f"{case['id']}.{mode}"
    Path(f"{prefix}.stdout").write_text(completed.stdout, encoding="utf-8")
    Path(f"{prefix}.stderr").write_text(completed.stderr, encoding="utf-8")

    result = {
        "id": case["id"],
        "regime": case["regime"],
        "subtype": case.get("subtype"),
        "language": case["language"],
        "mode": mode,
        "returncode": completed.returncode,
        "wall_seconds": duration,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
    }
    result.update(parse_metrics(completed.stderr))
    if mode == "cache":
        result["cache_active"] = result["cache_hits"] + result["cache_misses"] > 0
    return result


def parse_args():
    default_corpus = Path(__file__).with_name("moe-cache-corpus.jsonl")
    parser = argparse.ArgumentParser(
        description="Run deterministic llama-debug comparisons with and without the persistent MoE cache."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/bin/llama-debug"))
    parser.add_argument("--corpus", type=Path, default=default_corpus)
    parser.add_argument("--output-dir", type=Path, default=Path("moe-cache-corpus-results"))
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--moe-cache-mib", type=int, default=2048)
    parser.add_argument("--mode", choices=("both", "baseline", "cache"), default="both")
    parser.add_argument("--regime", action="append", default=[])
    parser.add_argument("--id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.n_predict <= 0:
        parser.error("--n-predict must be positive")
    if args.moe_cache_mib <= 0:
        parser.error("--moe-cache-mib must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main():
    args = parse_args()
    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    if "--moe-cache-mib" in extra_args:
        raise ValueError("pass the cache size through --moe-cache-mib before --")

    cases = load_corpus(args.corpus)
    if args.regime:
        regimes = set(args.regime)
        cases = [case for case in cases if case["regime"] in regimes]
    if args.id:
        ids = set(args.id)
        cases = [case for case in cases if case["id"] in ids]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("no corpus entries selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = ("baseline", "cache") if args.mode == "both" else (args.mode,)
    by_id = {}
    failed = False

    for case in cases:
        by_id[case["id"]] = {
            "id": case["id"],
            "regime": case["regime"],
            "subtype": case.get("subtype"),
            "language": case["language"],
        }
        for mode in modes:
            print(f"{case['id']}: running {mode}", flush=True)
            result = run_case(args, case, mode, extra_args, args.output_dir)
            by_id[case["id"]][mode] = result
            failed = failed or result["returncode"] != 0
            failed = failed or (mode == "cache" and not result["cache_active"])

        comparison = by_id[case["id"]]
        if "baseline" in comparison and "cache" in comparison:
            baseline = comparison["baseline"]
            cache = comparison["cache"]
            output_match = baseline["stdout_sha256"] == cache["stdout_sha256"]
            comparison["output_match"] = output_match
            failed = failed or not output_match

            if baseline.get("tokens_per_second") and cache.get("tokens_per_second"):
                comparison["speedup"] = cache["tokens_per_second"] / baseline["tokens_per_second"]
            if baseline["selected_bytes"] > 0 and cache["cache_active"]:
                cache_traffic = cache["selected_bytes"] + cache["fill_bytes"]
                comparison["traffic_reduction"] = 1.0 - cache_traffic / baseline["selected_bytes"]

            speedup = comparison.get("speedup")
            reduction = comparison.get("traffic_reduction")
            speedup_text = f"{speedup:.3f}" if speedup is not None else "n/a"
            reduction_text = f"{reduction:.3f}" if reduction is not None else "n/a"
            print(
                f"{case['id']}: match={output_match} "
                f"speedup={speedup_text} traffic_reduction={reduction_text}",
                flush=True,
            )

    results = list(by_id.values())
    regimes = {}
    for result in results:
        regime = regimes.setdefault(result["regime"], {
            "regime": result["regime"],
            "cases": 0,
            "output_matches": 0,
            "speedups": [],
            "traffic_reductions": [],
        })
        regime["cases"] += 1
        regime["output_matches"] += int(result.get("output_match", False))
        if "speedup" in result:
            regime["speedups"].append(result["speedup"])
        if "traffic_reduction" in result:
            regime["traffic_reductions"].append(result["traffic_reduction"])
    for regime in regimes.values():
        speedups = regime.pop("speedups")
        reductions = regime.pop("traffic_reductions")
        regime["mean_speedup"] = sum(speedups) / len(speedups) if speedups else None
        regime["mean_traffic_reduction"] = sum(reductions) / len(reductions) if reductions else None

    summary = {
        "model": str(args.model),
        "corpus": str(args.corpus),
        "n_predict": args.n_predict,
        "moe_cache_mib": args.moe_cache_mib,
        "extra_args": extra_args,
        "regimes": [regimes[name] for name in sorted(regimes)],
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
