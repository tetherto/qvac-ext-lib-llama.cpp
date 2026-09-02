# llama.cpp/examples/debug

This is a utility intended to help debug a model by registering a callback that
logs GGML operations and tensor data. It can also store the generated logits or
embeddings as well as the prompt and token ids for comparison with the original
model.

### Usage

```shell
llama-debug \
  --hf-repo ggml-org/models \
  --hf-file phi-2/ggml-model-q4_0.gguf \
  --model phi-2-q4_0.gguf \
  --prompt hello \
  --save-logits \
  --verbose
```

Pass `--n-predict N` to greedily decode additional tokens and inspect tensors
from the token-generation graphs:

```shell
llama-debug \
  --model model.gguf \
  --prompt "Hello" \
  --n-predict 32 \
  --tensor-filter 'ffn_moe_topk' \
  --no-warmup \
  --verbose > trace.log 2>&1
```

The resulting MoE routing trace can be summarized with:

```shell
python3 examples/debug/analyze-moe-trace.py trace.log \
  --capacities 128,256,512,1024,2048
```

Multiple trace paths can be passed and are treated as independent cold-cache runs.
Set `GGML_MOE_TRACE=1` to print selected-expert copy bytes and ID readback time when testing the scheduler's selected-slice offload path.
Set `GGML_MOE_COMPACT=1` to use compact transient expert banks for single-token graphs.
Pass `--moe-cache-mib N` with CPU-resident MoE weights to use a context-owned persistent LRU cache.
`GGML_DEBUG_FULL=1` prints all values of matching tensors for numerical comparisons.

The MoE cache corpus contains 24 prompts covering prose, code, debugging, mathematics, structured output, dialogue, planning, translation, and mixed code with prose. The runner executes each prompt as an independent cold-cache workload in baseline and persistent-cache modes, compares deterministic generated output, and records decode speed and transfer telemetry:

```shell
python3 examples/debug/run-moe-cache-corpus.py \
  --model model.gguf \
  --binary build/bin/llama-debug \
  --n-predict 128 \
  --moe-cache-mib 2048 \
  -- \
  -dev CUDA0 -sm none -ngl 999 -cmoe --load-mode none -t 16 -fit off
```

Use `--regime code`, `--id cpp-lru`, or `--limit 2` for a smaller run. Raw output and a machine-readable `summary.json` with per-case and per-regime results are written to `moe-cache-corpus-results`.

For a larger corpus, use the persistent-server runner to load the model only once per mode rather than once per prompt:

```shell
python3 examples/debug/run-moe-cache-server-corpus.py \
  --model model.gguf \
  --binary build/bin/llama-server \
  --n-predict 128 \
  --moe-cache-mib 2048 \
  -- \
  -dev CUDA0 -sm none -ngl 999 -cmoe --load-mode none -t 16 -fit off
```

The server runner disables KV prompt reuse between requests while deliberately keeping the context-owned MoE cache alive across the corpus. `--mode both` starts one baseline server and one cache server, so the full comparison loads the model twice total. Use separate baseline/cache invocations when their tensor-placement arguments differ. Generated text hashes are recorded but do not fail the run unless `--require-output-match` is passed.

Set `GGML_MOE_PREFILL_OVERLAP=1` to try pipelined cache fills during multi-token prompt processing. This requires expert weights in the device's pinned host buffer (`--load-mode none`) and a cache large enough for two complete expert layers. Batches that select at least 95% of the experts use two full-layer buffers; less dense batches copy only selected experts and pipeline the projection copies. Override the full-layer threshold with `GGML_MOE_PREFILL_MIN_DENSITY` in the range `(0, 1]`.

This can be combined with `--fit`: fully resident expert layers continue to execute directly from GPU weights, complete host-resident layers use the cache, and a fractional placement-boundary layer uses ordinary offload. Full-layer prefill buffers borrow two physical cache ranges without becoming LRU entries, so the rest of the decode working set survives the transition. The most recently routed prompt experts are promoted from staging into the preserved LRU with device-to-device copies before decode.

The tensor data is logged as debug and requires the --verbose flag. The reason
for this is that while useful for a model with many layers there can be a lot of
output. You can filter the tensor names using the `--tensor-filter` option.

A recommended approach is to first run without `--verbose` and see if the
generated logits/embeddings are close to the original model. If they are not,
then it might be required to inspect tensor by tensor and in that case it is
useful to enable the `--verbose` flag along with `--tensor-filter` to focus on
specific tensors.

### Options
This example supports all standard `llama.cpp` options and also accepts the
following options:
```console
$ llama-debug --help
...

----- example-specific params -----

--save-logits                           save final logits to files for verification (default: false)
--logits-output-dir PATH                directory for saving logits output files (default: data)
--tensor-filter REGEX                   filter tensor names for debug output (regex pattern, can be specified multiple times)
```

### Output Files

When `--save-logits` is enabled, the following files are created in the output
directory:

* `llamacpp-<model>[-embeddings].bin`        - Binary output (logits or embeddings)
* `llamacpp-<model>[-embeddings].txt`        - Text output (logits or embeddings, one per line)
* `llamacpp-<model>[-embeddings]-prompt.txt` - Prompt text and token IDs
* `llamacpp-<model>[-embeddings]-tokens.bin` - Binary token IDs for programmatic comparison

These files can be compared against the original model's output to verify the
converted model.
