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

    std::vector<llama_token> tokens = common_tokenize(tctx, corpus, true);
    ggml_opt_dataset_t ds = common_opt_dataset_init(tctx, tokens, llama_n_ctx(tctx)/2);
    if (!ds) { err = "dataset init failed (corpus too short for ctx?)"; llama_free(tctx); return false; }

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
    srv.listen(host, port);
    return 0;
}
