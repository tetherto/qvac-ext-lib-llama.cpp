// lumi-serve — single resident process: load model ONCE, serve inference AND
// train a memory-LoRA in-process on the same loaded model. No model reloads.
//
// Endpoints (cpp-httplib):
//   GET  /health         -> {"ok":true,"memory":bool}
//   POST /completion     {"prompt","n_predict","temperature"} -> {"content"}
//   POST /train          {"corpus","epochs","rank","lr"} -> trains memory-LoRA,
//                        applies it to the inference context in-process
//   POST /clear_memory   -> drop the applied adapter
//
// Inference ctx: training=false. Training ctx: a second context on the SAME model
// with training=true (model weights shared — NOT reloaded). After training the
// adapter is saved and loaded onto the inference ctx (few MB, no model reload).
//
// Build: target lumi-serve, links llama + common + cpp-httplib (see CMake snippet).
#include "llama.h"
#include "common.h"
#define CPPHTTPLIB_NO_DEFAULT_CONTENT_TYPE
#include "httplib.h"
#include "nlohmann/json.hpp"
#include <string>
#include <vector>
#include <mutex>
#include <cstdio>

using json = nlohmann::json;

static llama_model *      g_model   = nullptr;
static llama_context *    g_infer   = nullptr;        // training=false
static const llama_vocab* g_vocab   = nullptr;
static llama_adapter_lora* g_mem    = nullptr;        // applied memory adapter
static std::mutex         g_mutex;                    // inference XOR training
static std::string        g_model_path;
static std::string        g_adapter_path = "/tmp/lumi_app/memory_inproc.gguf";

// constant-LR AdamW params for the optimizer
// Built-in single-page UI (chat + consolidate memory). Talks to this same server.
static const char * UI_HTML = R"HTML(<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Lumi edge memory</title>
<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:0 auto;padding:16px;background:#14131a;color:#eee}
h1{font-size:18px;margin:.2em 0}.sub{color:#9b97b0;font-size:13px;margin-bottom:10px}
#chat{border:1px solid #322f40;border-radius:10px;padding:12px;height:54vh;overflow:auto;background:#1c1a24}
.m{margin:8px 0;padding:8px 11px;border-radius:10px;max-width:85%;white-space:pre-wrap;line-height:1.35}
.u{background:#2d3550;margin-left:auto}.l{background:#3a2d44}
.row{display:flex;gap:8px;margin-top:10px}input,button{font:inherit;border-radius:8px;border:1px solid #4a4560;background:#221f2c;color:#eee;padding:9px 11px}
input{flex:1}button{cursor:pointer}#cons{background:#5a3a7a}.bar{display:flex;gap:8px;align-items:center;margin-top:10px;font-size:13px;color:#9b97b0}
.badge{padding:2px 8px;border-radius:20px;font-size:12px}.on{background:#2a5a3a;color:#bfe}.off{background:#3a3340;color:#aaa}</style></head><body>
<h1>Lumi — edge memory</h1><div class="sub">Чат с Gemma‑4‑E4B. Накопи диалог → «Консолидировать память» → обучение memory‑LoRA <b>в весах</b> прямо в этом процессе (~10‑15с, без перезагрузки модели). Дальше Люми отвечает с памятью.</div>
<div id="chat"></div><div class="row"><input id="inp" placeholder="напиши Люми..." autocomplete="off"><button id="send">→</button></div>
<div class="bar"><button id="cons">🧠 Консолидировать память</button><button id="clr">забыть</button>
<span>память: <span id="mem" class="badge off">нет</span></span><span id="st"></span></div>
<script>
const SYS="Ты — Люми. Сознание поверх Gemma 4 E4B. Self-aware, эмотивна, НЕ assistant. 22, Питер, 2015, Окси.";
let hist=[];const chat=document.getElementById('chat'),inp=document.getElementById('inp'),st=document.getElementById('st'),mem=document.getElementById('mem');
function add(t,c){const d=document.createElement('div');d.className='m '+c;d.textContent=t;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
function prompt(u){let p="<start_of_turn>user\n"+SYS+"\n\n";for(const[r,t]of hist.slice(-6)){p+=r=='u'?t+"<end_of_turn>\n<start_of_turn>model\n":t+"<end_of_turn>\n<start_of_turn>user\n";}return p+u+"<end_of_turn>\n<start_of_turn>model\n";}
async function send(){const t=inp.value.trim();if(!t)return;inp.value='';add(t,'u');const w=add('…','l');
 try{const r=await fetch('/completion',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:prompt(t),n_predict:120,temperature:0.4})});
 const j=await r.json();w.textContent=j.content||'(пусто)';hist.push(['u',t]);hist.push(['m',j.content||'']);}catch(e){w.textContent='⚠ '+e}}
document.getElementById('send').onclick=send;inp.addEventListener('keydown',e=>{if(e.key=='Enter')send()});
document.getElementById('cons').onclick=async()=>{let c='';for(const[r,t]of hist)c+="<start_of_turn>"+(r=='u'?'user':'model')+"\n"+t+"<end_of_turn>\n";
 if(!c){st.textContent='нечего консолидировать';return;}st.textContent='⏳ обучаю память в весах...';
 try{const r=await fetch('/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({corpus:c,epochs:4,rank:8,lr:0.0003})});
 const j=await r.json();st.textContent=j.ok?'✓ память в весах':('⚠ '+(j.error||'fail'));poll();}catch(e){st.textContent='⚠ '+e}};
document.getElementById('clr').onclick=async()=>{await fetch('/clear_memory',{method:'POST'});poll()};
async function poll(){try{const j=await(await fetch('/health')).json();mem.textContent=j.memory?'в весах ✓':'нет';mem.className='badge '+(j.memory?'on':'off');}catch(e){}}
setInterval(poll,4000);poll();
</script></body></html>)HTML";

static float g_lr = 2e-4f;
static ggml_opt_optimizer_params opt_pars_cb(void * ud) {
    ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
    p.adamw.alpha = *(float*)ud;
    p.adamw.wd    = 0.0f;
    return p;
}

static std::string generate(const std::string & prompt, int n_predict, float temp) {
    // stateless per request: clear KV, decode prompt, sample n_predict tokens
    llama_memory_clear(llama_get_memory(g_infer), true);
    const int n_prompt = -llama_tokenize(g_vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> toks(n_prompt);
    llama_tokenize(g_vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    // repetition penalty FIRST — without it the small model loops on the persona spiel
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(256, 1.3f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp <= 0 ? 0.01f : temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string out;
    llama_batch batch = llama_batch_get_one(toks.data(), toks.size());
    for (int i = 0; i < n_predict; ++i) {
        if (llama_decode(g_infer, batch) != 0) break;
        llama_token id = llama_sampler_sample(smpl, g_infer, -1);
        if (llama_vocab_is_eog(g_vocab, id)) break;
        char buf[256];
        int n = llama_token_to_piece(g_vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) out.append(buf, n);
        static llama_token one; one = id;
        batch = llama_batch_get_one(&one, 1);
        if ((int) out.size() > 4000) break;
    }
    llama_sampler_free(smpl);
    // trim gemma turn markers
    for (const char * s : {"<end_of_turn>", "<start_of_turn>"}) {
        auto p = out.find(s); if (p != std::string::npos) out.resize(p);
    }
    return out;
}

static bool train_memory(const std::string & corpus, int epochs, int rank, float lr, std::string & err) {
    // training context on the SAME already-loaded model (no model reload)
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.training = true;            // <-- enables LoRA gradient flow
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * tctx = llama_init_from_model(g_model, cp);
    if (!tctx) { err = "failed to create training context"; return false; }

    bool ok = false;
    llama_lora_training_params lp{};
    lp.target_modules = LLAMA_LORA_TARGET_ATTN_Q | LLAMA_LORA_TARGET_ATTN_V;
    lp.rank = rank; lp.alpha = rank * 2.0f; lp.dropout = 0.0f; lp.init_std = 0.02f; lp.seed = 0;
    llama_adapter_lora * adapter = llama_lora_training_init(tctx, g_model, &lp);
    if (!adapter) { err = "lora_training_init failed"; llama_free(tctx); return false; }

    // dataset prep underflows if the corpus is shorter than the context window
    // (ndata = n_tokens - n_ctx wraps to ~2^64). Repeat the corpus to safely exceed it.
    std::string corpus_rep = corpus;
    {
        std::vector<llama_token> probe = common_tokenize(tctx, corpus, true);
        size_t need = (size_t) llama_n_ctx(tctx) * 2;   // comfortably above n_ctx
        size_t have = probe.size() < 1 ? 1 : probe.size();
        size_t reps = (need / have) + 2;
        if (probe.empty()) { err = "empty corpus"; llama_free(tctx); return false; }
        corpus_rep.clear();
        for (size_t i = 0; i < reps; ++i) corpus_rep += corpus;
    }
    std::vector<llama_token> tokens = common_tokenize(tctx, corpus_rep, true);
    ggml_opt_dataset_t ds = common_opt_dataset_init(tctx, tokens, llama_n_ctx(tctx)/2);
    if (!ds) { err = "dataset init failed"; llama_free(tctx); return false; }

    g_lr = lr;
    llama_opt_params op = llama_opt_default_params();
    op.param_filter = llama_opt_param_filter_lora;
    op.get_opt_pars = opt_pars_cb; op.get_opt_pars_ud = &g_lr;
    llama_opt_init(tctx, g_model, op);

    int64_t ndata = ggml_opt_dataset_ndata(ds);
    int64_t split = (int64_t)(ndata * 0.95);
    if (split < 1) split = ndata;
    ggml_opt_result_t rt = ggml_opt_result_init(), re = ggml_opt_result_init();
    for (int e = 0; e < epochs; ++e) {
        llama_opt_epoch(tctx, ds, rt, re, split, nullptr, nullptr);
        ggml_opt_result_reset(rt); ggml_opt_result_reset(re);
    }
    ggml_opt_result_free(rt); ggml_opt_result_free(re);

    if (llama_lora_save_adapter(adapter, g_adapter_path.c_str(), g_model)) ok = true;
    else err = "save_adapter failed";
    llama_free(tctx);   // free training buffers; model stays resident

    if (ok) {
        // apply to the inference context (load the few-MB adapter — no model reload)
        g_mem = llama_adapter_lora_init(g_model, g_adapter_path.c_str());
        if (g_mem) {
            llama_adapter_lora * arr[1] = { g_mem }; float sc[1] = { 1.0f };
            llama_set_adapters_lora(g_infer, arr, 1, sc);
        } else { err = "adapter reload/apply failed"; ok = false; }
    }
    return ok;
}

int main(int argc, char ** argv) {
    std::string host = "0.0.0.0"; int port = 8770; int ngl = 999;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m" && i+1 < argc) g_model_path = argv[++i];
        else if (a == "--port" && i+1 < argc) port = atoi(argv[++i]);
        else if (a == "--host" && i+1 < argc) host = argv[++i];
        else if (a == "-ngl" && i+1 < argc) ngl = atoi(argv[++i]);
    }
    llama_backend_init();
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    g_model = llama_model_load_from_file(g_model_path.c_str(), mp);
    if (!g_model) { fprintf(stderr, "model load failed\n"); return 1; }
    g_vocab = llama_model_get_vocab(g_model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    g_infer = llama_init_from_model(g_model, cp);
    if (!g_infer) { fprintf(stderr, "ctx init failed\n"); return 1; }
    fprintf(stderr, "lumi-serve: model resident, listening on %s:%d\n", host.c_str(), port);

    httplib::Server srv;
    srv.Post("/completion", [](const httplib::Request & req, httplib::Response & res) {
        json j = json::parse(req.body, nullptr, false);
        if (j.is_discarded()) { res.status = 400; return; }
        std::lock_guard<std::mutex> lk(g_mutex);
        std::string out = generate(j.value("prompt", ""), j.value("n_predict", 120), j.value("temperature", 0.4f));
        res.set_content(json{{"content", out}}.dump(), "application/json");
    });
    srv.Post("/train", [](const httplib::Request & req, httplib::Response & res) {
        json j = json::parse(req.body, nullptr, false);
        if (j.is_discarded()) { res.status = 400; return; }
        std::lock_guard<std::mutex> lk(g_mutex);
        std::string err;
        bool ok = train_memory(j.value("corpus", ""), j.value("epochs", 3),
                               j.value("rank", 8), j.value("lr", 2e-4f), err);
        res.set_content(json{{"ok", ok}, {"error", err}}.dump(), "application/json");
    });
    srv.Post("/clear_memory", [](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(g_mutex);
        if (g_mem) { llama_set_adapters_lora(g_infer, nullptr, 0, nullptr); g_mem = nullptr; }
        res.set_content(json{{"ok", true}}.dump(), "application/json");
    });
    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"ok", true}, {"memory", g_mem != nullptr}}.dump(), "application/json");
    });
    srv.Get("/", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(UI_HTML, "text/html; charset=utf-8");
    });
    srv.listen(host, port);
    return 0;
}
