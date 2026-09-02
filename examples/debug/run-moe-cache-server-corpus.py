#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TELEMETRY_RE = re.compile(
    r"moe selected bytes = (?P<selected>\d+).*?"
    r"cache = (?P<hits>\d+)/(?P<misses>\d+) hits/misses, "
    r"(?P<evictions>\d+) evictions, (?P<fill>\d+) fill bytes"
)
ID_RE = re.compile(r"^[a-z0-9-]+$")
SERVER_OWNED_ARGS = {
    "-m",
    "--model",
    "--host",
    "--port",
    "-np",
    "--parallel",
    "--moe-cache-mib",
}


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


def parse_telemetry(log_text):
    telemetry = [match.groupdict() for match in TELEMETRY_RE.finditer(log_text)]
    hits = sum(int(item["hits"]) for item in telemetry)
    misses = sum(int(item["misses"]) for item in telemetry)
    result = {
        "telemetry_records": len(telemetry),
        "selected_bytes": sum(int(item["selected"]) for item in telemetry),
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_evictions": sum(int(item["evictions"]) for item in telemetry),
        "fill_bytes": sum(int(item["fill"]) for item in telemetry),
    }
    if hits + misses > 0:
        result["cache_hit_rate"] = hits / (hits + misses)
    return result


def select_cases(args):
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
    if args.combine_prompts:
        cases = [{
            "id": "combined-prefill",
            "regime": "mixed",
            "subtype": "combined",
            "language": "multi",
            "prompt": "\n\n".join(case["prompt"] for case in cases),
        }]
    return cases


def pick_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request_json(url, payload=None, timeout=5):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def tail_text(path, limit=40):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:])


class PersistentServer:
    def __init__(self, args, mode, extra_args, output_dir):
        self.args = args
        self.mode = mode
        self.port = pick_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = output_dir / f"server.{mode}.log"
        self.log_handle = None
        self.log_offset = 0
        self.process = None
        self.command = [
            str(args.binary),
            "--model", str(args.model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--parallel", "1",
            "--no-webui",
            "--no-warmup",
        ]
        self.command.extend(extra_args)
        if mode == "cache":
            self.command.extend(["--moe-cache-mib", str(args.moe_cache_mib)])

    def start(self):
        env = os.environ.copy()
        env["GGML_MOE_TRACE"] = "1"
        env.setdefault("GGML_OP_OFFLOAD_MIN_BATCH", "1")
        env.pop("GGML_MOE_COMPACT", None)
        env.pop("GGML_DEBUG_FULL", None)
        env.pop("LLAMA_ARG_MOE_CACHE_MIB", None)

        self.log_handle = self.log_path.open("wb")
        started = time.monotonic()
        self.process = subprocess.Popen(
            self.command,
            env=env,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
        )
        deadline = started + self.args.startup_timeout
        last_error = None
        while time.monotonic() < deadline:
            returncode = self.process.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"llama-server exited with status {returncode} during startup\n"
                    f"{tail_text(self.log_path)}"
                )
            try:
                request_json(f"{self.url}/health", timeout=1)
                self.log_offset = self.log_path.stat().st_size
                return time.monotonic() - started
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                time.sleep(0.25)
        raise RuntimeError(
            f"llama-server was not healthy after {self.args.startup_timeout:.1f} seconds: {last_error}\n"
            f"{tail_text(self.log_path)}"
        )

    def read_new_log(self):
        with self.log_path.open("rb") as log_file:
            log_file.seek(self.log_offset)
            data = log_file.read()
            self.log_offset = log_file.tell()
        return data.decode("utf-8", errors="replace")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        if self.log_handle is not None:
            self.log_handle.close()


def run_case(args, case, mode, server, output_dir):
    payload = {
        "prompt": "\n\n".join([case["prompt"]] * args.prompt_repeat),
        "n_predict": args.n_predict,
        "temperature": 0,
        "seed": args.seed,
        "ignore_eos": True,
        "cache_prompt": False,
        "add_special": True,
        "stream": False,
    }
    started = time.monotonic()
    response = request_json(f"{server.url}/completion", payload, timeout=args.request_timeout)
    duration = time.monotonic() - started
    telemetry = parse_telemetry(server.read_new_log())

    content = response.get("content")
    timings = response.get("timings")
    if not isinstance(content, str):
        raise ValueError(f"{case['id']}: completion response has no string content")
    if not isinstance(timings, dict):
        raise ValueError(f"{case['id']}: completion response has no timings object")

    prefix = output_dir / f"{case['id']}.{mode}"
    Path(f"{prefix}.stdout").write_text(content, encoding="utf-8")
    Path(f"{prefix}.response.json").write_text(
        json.dumps(response, indent=2) + "\n", encoding="utf-8"
    )

    result = {
        "id": case["id"],
        "regime": case["regime"],
        "subtype": case.get("subtype"),
        "language": case["language"],
        "mode": mode,
        "returncode": 0,
        "wall_seconds": duration,
        "stdout_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "prompt_tokens": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_tokens_per_second": timings.get("prompt_per_second"),
        "eval_runs": timings.get("predicted_n"),
        "eval_ms": timings.get("predicted_ms"),
        "tokens_per_second": timings.get("predicted_per_second"),
        "tokens_cached": response.get("tokens_cached"),
    }
    result.update(telemetry)
    if mode == "cache":
        result["cache_active"] = result["cache_hits"] + result["cache_misses"] > 0
    return result


def validate_extra_args(extra_args):
    for value in extra_args:
        option = value.split("=", 1)[0]
        if option in SERVER_OWNED_ARGS:
            raise ValueError(f"{option} is managed by the corpus runner")


def add_comparisons(results, require_output_match):
    failed = False
    for comparison in results:
        if "baseline" not in comparison or "cache" not in comparison:
            continue
        baseline = comparison["baseline"]
        cache = comparison["cache"]
        output_match = baseline["stdout_sha256"] == cache["stdout_sha256"]
        comparison["output_match"] = output_match
        failed = failed or (require_output_match and not output_match)

        if baseline.get("tokens_per_second") and cache.get("tokens_per_second"):
            comparison["speedup"] = cache["tokens_per_second"] / baseline["tokens_per_second"]
        if baseline["selected_bytes"] > 0 and cache.get("cache_active"):
            cache_traffic = cache["selected_bytes"] + cache["fill_bytes"]
            comparison["traffic_reduction"] = 1.0 - cache_traffic / baseline["selected_bytes"]

        speedup = comparison.get("speedup")
        hit_rate = cache.get("cache_hit_rate")
        speedup_text = f"{speedup:.3f}" if speedup is not None else "n/a"
        hit_rate_text = f"{hit_rate:.3f}" if hit_rate is not None else "n/a"
        print(
            f"{comparison['id']}: match={output_match} "
            f"speedup={speedup_text} hit_rate={hit_rate_text}",
            flush=True,
        )
    return failed


def summarize_regimes(results):
    regimes = {}
    for result in results:
        regime = regimes.setdefault(result["regime"], {
            "regime": result["regime"],
            "cases": 0,
            "output_matches": 0,
            "speedups": [],
            "traffic_reductions": [],
            "cache_hit_rates": [],
        })
        regime["cases"] += 1
        regime["output_matches"] += int(result.get("output_match", False))
        if "speedup" in result:
            regime["speedups"].append(result["speedup"])
        if "traffic_reduction" in result:
            regime["traffic_reductions"].append(result["traffic_reduction"])
        cache = result.get("cache")
        if cache and "cache_hit_rate" in cache:
            regime["cache_hit_rates"].append(cache["cache_hit_rate"])

    for regime in regimes.values():
        for values_key, mean_key in (
            ("speedups", "mean_speedup"),
            ("traffic_reductions", "mean_traffic_reduction"),
            ("cache_hit_rates", "mean_cache_hit_rate"),
        ):
            values = regime.pop(values_key)
            regime[mean_key] = sum(values) / len(values) if values else None
    return [regimes[name] for name in sorted(regimes)]


def parse_args():
    default_corpus = Path(__file__).with_name("moe-cache-corpus.jsonl")
    parser = argparse.ArgumentParser(
        description="Run a corpus through one persistent llama-server process per MoE-cache mode."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/bin/llama-server"))
    parser.add_argument("--corpus", type=Path, default=default_corpus)
    parser.add_argument("--output-dir", type=Path, default=Path("moe-cache-server-corpus-results"))
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help="Repeat each corpus prompt N times to exercise long-prefill behavior.",
    )
    parser.add_argument(
        "--combine-prompts",
        action="store_true",
        help="Join all selected corpus prompts into one diverse prefill request.",
    )
    parser.add_argument("--moe-cache-mib", type=int, default=2048)
    parser.add_argument("--mode", choices=("both", "baseline", "cache"), default="both")
    parser.add_argument("--regime", action="append", default=[])
    parser.add_argument("--id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--require-output-match", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.n_predict <= 0:
        parser.error("--n-predict must be positive")
    if args.prompt_repeat <= 0:
        parser.error("--prompt-repeat must be positive")
    if args.moe_cache_mib <= 0:
        parser.error("--moe-cache-mib must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")
    return args


def main():
    args = parse_args()
    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    validate_extra_args(extra_args)

    cases = select_cases(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = ("baseline", "cache") if args.mode == "both" else (args.mode,)
    by_id = {
        case["id"]: {
            "id": case["id"],
            "regime": case["regime"],
            "subtype": case.get("subtype"),
            "language": case["language"],
        }
        for case in cases
    }
    server_runs = {}
    failed = False

    for mode in modes:
        server = PersistentServer(args, mode, extra_args, args.output_dir)
        print(f"{mode}: starting persistent server", flush=True)
        try:
            startup_seconds = server.start()
            print(f"{mode}: server ready after {startup_seconds:.2f}s", flush=True)
            server_runs[mode] = {
                "startup_seconds": startup_seconds,
                "command": server.command,
                "log": str(server.log_path),
            }
            for index, case in enumerate(cases, 1):
                print(f"{mode}: [{index}/{len(cases)}] {case['id']}", flush=True)
                result = run_case(args, case, mode, server, args.output_dir)
                by_id[case["id"]][mode] = result
                failed = failed or (mode == "cache" and not result["cache_active"])
        finally:
            server.stop()

    results = list(by_id.values())
    failed = add_comparisons(results, args.require_output_match) or failed
    summary = {
        "model": str(args.model),
        "corpus": str(args.corpus),
        "n_predict": args.n_predict,
        "prompt_repeat": args.prompt_repeat,
        "combine_prompts": args.combine_prompts,
        "moe_cache_mib": args.moe_cache_mib,
        "extra_args": extra_args,
        "persistent_across_cases": True,
        "server_runs": server_runs,
        "regimes": summarize_regimes(results),
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, HTTPError, URLError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
