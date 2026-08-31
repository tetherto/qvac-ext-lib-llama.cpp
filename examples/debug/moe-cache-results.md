# MoE Cache Test Results

This is an append-only record of persistent MoE cache correctness and performance tests.

## 2026-08-31: Persistent cache baseline

### Configuration

- Model: Qwen3.6-35B-A3B-MTP Q8_0
- GPU: NVIDIA GeForce RTX 5090
- CPU: AMD EPYC 7742, physical CPUs 16-31, NUMA node 1
- Expert weights: CPU resident
- GPU cache budget: 2048 MiB
- Allocated cache: 2046.38 MiB, 641 complete expert slots
- Decode offload threshold: `GGML_OP_OFFLOAD_MIN_BATCH=1`

### Backend and policy tests

| Test | Result |
| --- | ---: |
| MoE cache LRU policy tests | passed |
| CPU inactive `MUL_MAT_ID` routes | 13/13 |
| CUDA inactive `MUL_MAT_ID` routes | 13/13 |
| CUDA active `MUL_MAT_ID` routes | 945/945 |
| CUDA `MUL_MAT_ID` fusion | 13/13 |

The policy tests cover cold fills, repeat hits, duplicate IDs, global cross-layer eviction, current-plan pinning, reset, and insufficient capacity.

### Numerical correctness

| Model and condition | Compared outputs | NMSE | Maximum absolute error |
| --- | ---: | ---: | ---: |
| Local GPT-OSS-20B MXFP4, scheduler rebuilt between tokens | 3 | 0.0 | 0.0 |
| Qwen3.6-35B-A3B Q8_0, scheduler rebuilt between tokens | 3 | 0.0 | 0.0 |

The persistent cache remained resident across scheduler rebuilds. Q4_K kernel and fusion coverage passed, but a whole-model Q4_K_M file was not available.

### Qwen repeated-token benchmark

This benchmark uses `llama-bench` generation with one warm-up token and retains cache state across repetitions.

| Mode | Throughput |
| --- | ---: |
| Selected-slice baseline | 22.231 t/s |
| Persistent 2048 MiB cache | 42.459 t/s |

The warm-cache throughput improvement was 91.0%. A cold single-token run measured 10.51 t/s without the cache and 9.72 t/s with the cache, a 7.5% cold-fill penalty.

### Mixed-regime corpus

The corpus contains 24 prompts across 12 regimes. Each prompt was run in a fresh baseline process and a fresh persistent-cache process with no warm-up and 128 predicted tokens.

```shell
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
numactl --physcpubind=16-31 --membind=1 \
python3 examples/debug/run-moe-cache-corpus.py \
  --model Qwen3.6-35B-A3B-MTP-Q8_0.gguf \
  --binary build-moe/bin/llama-debug \
  --n-predict 128 \
  --moe-cache-mib 2048 \
  -- \
  -dev CUDA0 -sm none -ngl 999 -cmoe --load-mode none -t 16 -c 1024 -fit off
```

All 24 cached generated token streams matched their uncached baselines exactly.

| Regime | Cases | Mean speedup | Mean transfer reduction |
| --- | ---: | ---: | ---: |
| Prose | 5 | 1.47x | 39.6% |
| Code | 5 | 1.57x | 44.3% |
| Debugging | 2 | 1.67x | 48.3% |
| Math | 2 | 1.65x | 48.1% |
| Structured | 2 | 1.42x | 37.4% |
| Dialogue | 1 | 1.44x | 38.1% |
| Policy | 1 | 1.51x | 36.4% |
| Planning | 1 | 1.50x | 41.9% |
| Instructions | 1 | 1.48x | 39.7% |
| Translation | 2 | 1.69x | 48.2% |
| Creative | 1 | 1.51x | 41.6% |
| Mixed code and prose | 1 | 1.64x | 47.4% |

Across all cases:

- Median throughput increased from 22.29 t/s to 33.21 t/s, a 49.0% improvement.
- Mean per-case speedup was 1.55x.
- Mean host-to-device transfer reduction was 42.9%.
- Mean expert cache hit rate was 46.1%.

Three initial timing samples were affected by system contention and were rerun individually. The corrected reruns were 1.49x for history prose, 1.45x for Spanish translation, and 1.51x for constrained poetry. Routing, transfer counts, and generated outputs were unchanged.
