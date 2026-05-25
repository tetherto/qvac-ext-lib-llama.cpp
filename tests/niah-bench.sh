#!/usr/bin/env bash
#
# NIAH (Needle In A Haystack, gkamradt-style) benchmark for KV cache
# quantization quality.
#
# Sweeps a 2-D grid of (context_length, depth_percent) cells: a single
# "needle" sentence is inserted at a known depth into a long Paul Graham
# haystack, and the model is asked to retrieve it. Score per cell is
# binary recall (1 if the answer keyword appears in the model's output).
#
# Output is the canonical NIAH heatmap as a (ctx × depth) text grid plus
# a CSV row per cell. The orchestrator (test-kv-cache-niah.py) stacks
# these per-cell heatmaps across cache configs so you can see exactly
# where TBQ / PQ / q4_0 start to lose retrieval at each context length.
#
# Why a separate bench (not folded into RULER)? RULER's `niah_single_*`
# evaluate retrieval at one fixed depth (50% of the haystack) and vary
# along orthogonal axes (needle type, multi-key, multi-query); the
# canonical NIAH benchmark from the Google blog post explicitly sweeps
# depth × ctx and renders the heatmap that's commonly cited as
# "NIAH performance". Folding into RULER would lose the depth axis.
#
# Usage (direct):
#   ./tests/niah-bench.sh -m model.gguf [options]
#
# Usage (sourced):
#   source tests/niah-bench.sh
#   niah_bench -m model.gguf -ctk tbq3_0 -ctv pq3_0 -ngl 99 -fa 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NH_DIR="${SCRIPT_DIR}/niah"
NH_VENV="${NH_DIR}/.venv"
NH_PREPARE="${SCRIPT_DIR}/niah-prepare.py"
NH_SCORE="${SCRIPT_DIR}/niah-score.py"

# Re-use RULER's Paul Graham essay dump if it exists — saves a second
# network fetch and means NIAH inherits whatever fallback corpus
# RULER's PG-download step ended up with.
NH_RULER_HAYSTACK="${SCRIPT_DIR}/ruler/scripts/data/synthetic/json/PaulGrahamEssays.json"

NH_DATA_DIR=""
NH_SERVER_PORT=""
NH_SERVER_PID=""
NH_SERVER_URL=""
NH_SERVER_LOG=""

# Canonical gkamradt grid (5 ctx × 7 depths = 35 cells). Each cell is one
# inference; the whole grid runs in ~3-5 minutes on Llama-3.1-8B Q4_K_M
# with f16 KV cache. Override with --ctx-lengths / --depth-percents.
NH_DEFAULT_CTX="2048 4096 8192 16384 32768"
NH_DEFAULT_DEPTHS="0 25 50 75 100"

niah_usage() {
    cat <<'EOF'
NIAH (Needle In A Haystack) benchmark for KV cache quantization quality

Required:
  -m, --model PATH          Path to GGUF model

Options:
  -ctk  TYPE                K cache type                        (default: f16)
  -ctv  TYPE                V cache type                        (default: f16)
  --ctx-lengths "N ..."     Context lengths to sweep            (default: "2048 4096 8192 16384 32768")
                            The largest must fit in the server's -c budget;
                            we set -c = max(ctx_lengths) + 256.
  --depth-percents "D ..."  Needle depths (0..100)              (default: "0 25 50 75 100")
  --tokenizer PATH          HF tokenizer name/path              (default: auto-detect)
  --server-bin PATH         Path to llama-server binary         (default: build/bin/llama-server)
  --cli-bin PATH            Deprecated alias for --server-bin.
  --needle "TEXT"           Override the needle sentence
                            (default: Greg Kamradt's San Francisco sandwich)
  --question "TEXT"         Override the retrieval question
  --answer-keys "K1|K2|..." Pipe-separated answer-key substrings (default: "dolores park|sandwich and sit")
  --haystack-path PATH      Use a custom haystack JSON/text file
  -q, --quiet               Suppress per-sample output
  --csv FILE                Write per-cell CSV results
  --log FILE                Write full log to file (auto-generated if --csv is set)
  -h, --help                Show this help

All other flags are forwarded to llama-server.

Example:
  ./tests/niah-bench.sh -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
      -ctk tbq3_0 -ctv pq3_0 -ngl 99 -fa 1
EOF
}

_nh_detect_tokenizer() {
    local model_path=$1
    local base
    base=$(basename "$model_path" | tr '[:upper:]' '[:lower:]')
    case "$base" in
        *llama-3.1-8b*|*llama-3-8b*|*llama-3.1*|*meta-llama*)
            echo "NousResearch/Meta-Llama-3.1-8B-Instruct" ;;
        *ministral*)
            echo "mistralai/Ministral-8B-Instruct-2410" ;;
        *mistral*)
            echo "mistralai/Mistral-7B-Instruct-v0.3" ;;
        *qwen2.5*)
            echo "Qwen/Qwen2.5-7B-Instruct" ;;
        *qwen*)
            echo "Qwen/Qwen2-7B-Instruct" ;;
        *)
            echo "NousResearch/Meta-Llama-3.1-8B-Instruct" ;;
    esac
}

# ── venv + deps ─────────────────────────────────────────────────────────────
_nh_ensure_uv() {
    local uv_dir="${NH_DIR}/.uv"
    local uv_bin="${uv_dir}/uv"
    if [[ -x "$uv_bin" ]]; then return 0; fi
    UV_INSTALL_DIR="$uv_dir" curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3
    [[ -x "$uv_bin" ]]
}

# See longbench-bench.sh for the flock rationale.
_nh_ensure_venv() {
    mkdir -p "${NH_DIR}"
    local lock="${NH_DIR}/.venv.lock"
    (
        flock -x 9
        if [[ -f "${NH_VENV}/bin/activate" ]]; then exit 0; fi
        echo "Creating Python venv at ${NH_VENV} ..."
        local uv_bin="${NH_DIR}/.uv/uv"
        if [[ "${NH_USE_UV:-0}" == "1" ]]; then
            if command -v uv &>/dev/null; then uv venv "${NH_VENV}"
            elif _nh_ensure_uv; then "$uv_bin" venv "${NH_VENV}"
            else echo "ERROR: NH_USE_UV=1 but failed to install uv" >&2; exit 1; fi
        elif python3 -m venv "${NH_VENV}" 2>/dev/null; then :
        elif command -v uv &>/dev/null; then uv venv "${NH_VENV}"
        elif _nh_ensure_uv; then "$uv_bin" venv "${NH_VENV}"
        else echo "ERROR: cannot create a Python venv (no python3-venv, no uv)" >&2; exit 1; fi
    ) 9>"$lock"
    local rc=$?
    if (( rc != 0 )); then return $rc; fi
    # shellcheck disable=SC1091
    source "${NH_VENV}/bin/activate"
}

_nh_ensure_deps() {
    _nh_ensure_venv
    local missing=0
    for pkg in numpy transformers jinja2; do
        python3 -c "import ${pkg}" 2>/dev/null || missing=1
    done
    if (( missing )); then
        echo "Installing NIAH Python dependencies into venv ..."
        python3 -m pip install --quiet numpy transformers jinja2 2>&1 | tail -5
    fi
}

# ── server lifecycle ────────────────────────────────────────────────────────
_nh_pick_port() {
    awk -v seed="$$$(date +%N)" 'BEGIN { srand(seed); printf "%d\n", 20000 + int(rand() * 40000) }'
}

_nh_start_server() {
    local server_bin=$1 model=$2 ctk=$3 ctv=$4 n_ctx=$5
    shift 5
    local -a extra=("$@")
    NH_SERVER_PORT=$(_nh_pick_port)
    NH_SERVER_URL="http://127.0.0.1:${NH_SERVER_PORT}"
    NH_SERVER_LOG=$(mktemp)
    echo "Starting llama-server (port=${NH_SERVER_PORT}, n_ctx=${n_ctx}, K=${ctk}, V=${ctv}) ..."
    "$server_bin" \
        -m "$model" \
        -ctk "$ctk" -ctv "$ctv" \
        -c "$n_ctx" \
        --host 127.0.0.1 \
        --port "$NH_SERVER_PORT" \
        --no-webui \
        "${extra[@]}" \
        >"$NH_SERVER_LOG" 2>&1 &
    NH_SERVER_PID=$!
    local i
    for ((i=0; i<120; i++)); do
        if ! kill -0 "$NH_SERVER_PID" 2>/dev/null; then
            echo "ERROR: llama-server died during startup:" >&2
            tail -20 "$NH_SERVER_LOG" >&2
            return 1
        fi
        if curl -sf "${NH_SERVER_URL}/health" >/dev/null 2>&1; then
            echo "Server ready after ${i}s."
            return 0
        fi
        sleep 1
    done
    echo "ERROR: llama-server did not become healthy within 120s." >&2
    tail -20 "$NH_SERVER_LOG" >&2
    return 1
}

_nh_stop_server() {
    if [[ -n "$NH_SERVER_PID" ]] && kill -0 "$NH_SERVER_PID" 2>/dev/null; then
        kill "$NH_SERVER_PID" 2>/dev/null
        local i
        for ((i=0; i<5; i++)); do
            kill -0 "$NH_SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$NH_SERVER_PID" 2>/dev/null || true
    fi
    [[ -n "$NH_SERVER_LOG" && -f "$NH_SERVER_LOG" ]] && rm -f "$NH_SERVER_LOG"
}

# ── inference + scoring ─────────────────────────────────────────────────────
_nh_infer_single() {
    local tokens_to_gen=$1 prompt=$2
    local body
    body=$(jq -nc \
        --arg p "$prompt" \
        --argjson n "$tokens_to_gen" \
        '{prompt: $p, n_predict: $n, temperature: 0, cache_prompt: false, add_special: false}')
    local resp rc=0
    resp=$(curl -sf --max-time 600 \
        -X POST "${NH_SERVER_URL}/completion" \
        -H "Content-Type: application/json" \
        -d "$body") || rc=$?
    if (( rc != 0 )); then
        echo "ERROR: /completion request failed (curl rc=$rc)" >&2
        return 1
    fi
    jq -r '.content' <<<"$resp"
}

_nh_score() {
    local prediction=$1 references_json=$2
    local payload
    payload=$(python3 -c '
import json, sys
print(json.dumps({"prediction": sys.argv[1], "references": json.loads(sys.argv[2])}))' \
        "$prediction" "$references_json")
    python3 "$NH_SCORE" --stdin <<<"$payload"
}

niah_bench() {
    local server_bin="build/bin/llama-server"
    local model=""
    local ctk="f16"
    local ctv="f16"
    local ctx_lengths="$NH_DEFAULT_CTX"
    local depth_percents="$NH_DEFAULT_DEPTHS"
    local tokenizer=""
    local needle=""
    local question=""
    local answer_keys="dolores park|sandwich and sit"
    local haystack_path=""
    local quiet=0
    local csv_file=""
    local log_file=""
    local -a extra_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)        niah_usage; return 0 ;;
            -m|--model)       model="$2";          shift 2 ;;
            -ctk)             ctk="$2";            shift 2 ;;
            -ctv)             ctv="$2";            shift 2 ;;
            --ctx-lengths)    ctx_lengths="$2";    shift 2 ;;
            --depth-percents) depth_percents="$2"; shift 2 ;;
            --tokenizer)      tokenizer="$2";      shift 2 ;;
            --cli-bin|--server-bin) server_bin="$2"; shift 2 ;;
            --needle)         needle="$2";         shift 2 ;;
            --question)       question="$2";       shift 2 ;;
            --answer-keys)    answer_keys="$2";    shift 2 ;;
            --haystack-path)  haystack_path="$2";  shift 2 ;;
            -q|--quiet)       quiet=1;             shift ;;
            --csv)            csv_file="$2";       shift 2 ;;
            --log)            log_file="$2";       shift 2 ;;
            *)                extra_args+=("$1");  shift ;;
        esac
    done

    if [[ -z "$model" ]]; then
        echo "ERROR: --model is required" >&2; niah_usage >&2; return 1
    fi
    if [[ ! -x "$server_bin" ]]; then
        echo "ERROR: llama-server binary not found at '$server_bin'" >&2
        echo "       Build with: cmake --build build --target llama-server" >&2
        return 1
    fi
    if [[ -z "$tokenizer" ]]; then
        tokenizer=$(_nh_detect_tokenizer "$model")
        echo "Auto-detected tokenizer: $tokenizer"
    fi

    # Convert pipe-separated answer keys into a JSON array string (for the
    # JSONL records) and a Python-friendly argv list (for niah-prepare.py).
    local -a answer_keys_arr=()
    IFS='|' read -r -a answer_keys_arr <<<"$answer_keys"

    # Per-(tokenizer × grid × needle) data cache.
    local tok_slug="${tokenizer//\//_}"
    tok_slug="${tok_slug// /-}"
    local grid_slug
    grid_slug=$(echo "${ctx_lengths}__${depth_percents}" | tr ' ' '_')
    NH_DATA_DIR="${NH_DIR}/_data/${tok_slug}/${grid_slug}"
    mkdir -p "$NH_DATA_DIR"

    if [[ -z "$log_file" ]] && [[ -n "$csv_file" ]]; then
        log_file="${csv_file%.csv}.txt"
    fi
    if [[ -n "$log_file" ]]; then
        exec > >(tee -a "$log_file") 2>&1
    fi

    _nh_ensure_deps

    # Banner ---------------------------------------------------------------
    echo ""
    echo "=========================================="
    echo " NIAH (Needle In A Haystack) Benchmark"
    echo "=========================================="
    echo "  Model:        $(basename "$model")"
    echo "  K type:       $ctk"
    echo "  V type:       $ctv"
    echo "  ctx_lengths:  $ctx_lengths"
    echo "  depths (%):   $depth_percents"
    echo "  Tokenizer:    $tokenizer"
    [[ ${#extra_args[@]} -gt 0 ]] && echo "  Extra:        ${extra_args[*]}"
    echo "=========================================="
    echo ""

    # Prepare prompts --------------------------------------------------------
    local prep_file="${NH_DATA_DIR}/cells.jsonl"
    if [[ ! -s "$prep_file" ]]; then
        echo "Preparing NIAH (ctx × depth) cells ..."
        local -a prepare_cmd=(
            python3 "$NH_PREPARE"
            --tokenizer "$tokenizer"
            --ctx-lengths "$ctx_lengths"
            --depth-percents "$depth_percents"
            --output-file "$prep_file"
        )
        if [[ -n "$haystack_path" ]]; then
            prepare_cmd+=(--haystack-path "$haystack_path")
        fi
        if [[ -f "$NH_RULER_HAYSTACK" ]]; then
            prepare_cmd+=(--ruler-haystack-cache "$NH_RULER_HAYSTACK")
        fi
        if [[ -n "$needle" ]]; then
            prepare_cmd+=(--needle "$needle")
        fi
        if [[ -n "$question" ]]; then
            prepare_cmd+=(--question "$question")
        fi
        if (( ${#answer_keys_arr[@]} > 0 )); then
            prepare_cmd+=(--answer-keys "${answer_keys_arr[@]}")
        fi
        "${prepare_cmd[@]}" 2>&1 | tail -5
        if [[ ! -s "$prep_file" ]]; then
            echo "ERROR: niah-prepare.py produced no output" >&2
            return 1
        fi
    fi
    echo ""

    # Pick n_ctx as max(ctx_lengths) + slack (BOS / chat template overhead).
    local max_ctx
    max_ctx=$(echo "$ctx_lengths" | tr ' ' '\n' | sort -n | tail -1)
    local n_ctx=$(( max_ctx + 256 ))

    trap '_nh_stop_server' EXIT INT TERM
    _nh_start_server "$server_bin" "$model" "$ctk" "$ctv" "$n_ctx" "${extra_args[@]}" || return 1

    # Inference + scoring ----------------------------------------------------
    # We store scores in an associative array keyed by "<ctx>_<depth>" so we
    # can render the heatmap at the end without re-reading the JSONL.
    declare -A cell_score
    declare -A cell_pred
    local total=0 hits=0
    while IFS= read -r line; do
        local input_text references_json ctx depth max_gen
        input_text=$(jq -r '.input' <<<"$line")
        references_json=$(jq -c '.references' <<<"$line")
        ctx=$(jq -r '.ctx_len' <<<"$line")
        # niah-prepare.py writes depth_percent as a JSON float (e.g. 25.0).
        # The rendering loop iterates the user-supplied --depth-percents
        # string ("0 25 50 75 100"), which are bare ints. Canonicalise both
        # sides with awk '%g' so cell_score lookups match — otherwise the
        # heatmap shows "-" everywhere even though scoring succeeded.
        depth=$(awk -v d="$(jq -r '.depth_percent' <<<"$line")" 'BEGIN{printf "%g", d}')
        max_gen=$(jq -r '.max_gen' <<<"$line")
        if [[ -z "$input_text" ]]; then continue; fi

        local prediction
        prediction=$(_nh_infer_single "$max_gen" "$input_text") || {
            echo "ERROR: inference failed for ctx=$ctx depth=$depth" >&2
            continue
        }
        local score
        score=$(_nh_score "$prediction" "$references_json") || {
            echo "ERROR: scoring failed for ctx=$ctx depth=$depth" >&2
            continue
        }
        local key="${ctx}_${depth}"
        cell_score[$key]=$score
        cell_pred[$key]=$prediction
        total=$((total + 1))
        if [[ "$score" == "1.000000" ]]; then hits=$((hits + 1)); fi

        if (( quiet == 0 )); then
            local pct
            pct=$(awk "BEGIN { printf \"%.0f\", $score * 100 }")
            # UTF-8-safe 60-byte truncation — see longbench-bench.sh.
            local short_pred
            short_pred=$(printf '%s' "$prediction" | tr '\n' ' ' \
                | head -c 60 | iconv -c -f UTF-8 -t UTF-8)
            printf "  [ctx=%6d depth=%6s] %s%%  got=%s\n" \
                "$ctx" "$depth%" "$pct" "$short_pred"
        fi
    done < "$prep_file"

    # Heatmap ----------------------------------------------------------------
    echo ""
    echo "=========================================="
    local _bpw_k _bpw_v
    case "$ctk" in
        f16) _bpw_k=16.0 ;; q8_0) _bpw_k=8.5 ;; q4_0) _bpw_k=4.5 ;;
        tbq3_0) _bpw_k=4.25 ;; tbq4_0) _bpw_k=5.25 ;;
        pq3_0) _bpw_k=3.25 ;; pq4_0) _bpw_k=4.25 ;; *) _bpw_k="?" ;;
    esac
    case "$ctv" in
        f16) _bpw_v=16.0 ;; q8_0) _bpw_v=8.5 ;; q4_0) _bpw_v=4.5 ;;
        tbq3_0) _bpw_v=4.25 ;; tbq4_0) _bpw_v=5.25 ;;
        pq3_0) _bpw_v=3.25 ;; pq4_0) _bpw_v=4.25 ;; *) _bpw_v="?" ;;
    esac
    echo " NIAH heatmap: K=$ctk(${_bpw_k}bpw)  V=$ctv(${_bpw_v}bpw)  Model=$(basename "$model")"
    echo "=========================================="
    echo ""

    # Header row: depths.
    printf "%-8s" "ctx ↓"
    for d in $depth_percents; do
        printf " %5s%%" "$d"
    done
    printf "  %5s\n" "row-avg"

    if [[ -n "$csv_file" ]]; then
        echo "model,cache_k,cache_v,ctx_len,depth_percent,score" > "$csv_file"
    fi

    local model_base
    model_base=$(basename "$model")

    local grid_sum=0 grid_count=0
    for c in $ctx_lengths; do
        printf "%-8d" "$c"
        local row_sum=0 row_count=0
        for d in $depth_percents; do
            # Canonicalise depth to the same '%g' form used at write time
            # (see comment in the inference loop above).
            local d_canon
            d_canon=$(awk -v d="$d" 'BEGIN{printf "%g", d}')
            local key="${c}_${d_canon}"
            if [[ -n "${cell_score[$key]:-}" ]]; then
                local s=${cell_score[$key]}
                local s_pct
                s_pct=$(awk "BEGIN { printf \"%.0f\", $s * 100 }")
                printf " %5s%%" "$s_pct"
                row_sum=$(awk "BEGIN { printf \"%.6f\", $row_sum + $s }")
                row_count=$((row_count + 1))
                grid_sum=$(awk "BEGIN { printf \"%.6f\", $grid_sum + $s }")
                grid_count=$((grid_count + 1))
                if [[ -n "$csv_file" ]]; then
                    echo "${model_base},${ctk},${ctv},${c},${d},${s_pct}" >> "$csv_file"
                fi
            else
                printf " %6s" "-"
            fi
        done
        if (( row_count > 0 )); then
            local row_avg
            row_avg=$(awk "BEGIN { printf \"%.1f\", $row_sum / $row_count * 100 }")
            printf "  %5s%%\n" "$row_avg"
        else
            printf "  %5s\n" "-"
        fi
    done

    echo ""
    if (( grid_count > 0 )); then
        niah_global_score=$(awk "BEGIN { printf \"%.1f\", $grid_sum / $grid_count * 100 }")
    else
        niah_global_score="N/A"
    fi
    echo "  Grid average:    ${niah_global_score}%  (${hits}/${total} cells passed)"
    [[ -n "$csv_file" ]] && echo "  CSV: $csv_file"
    [[ -n "$log_file" ]] && echo "  Log: $log_file"
    echo "=========================================="
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -eo pipefail
    niah_bench "$@"
fi
