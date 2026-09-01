// Pins the CUDA arch-availability registration guard: devices with no compiled
// kernels for their compute capability are skipped in ggml_backend_cuda_reg(),
// and everything downstream must cope with the registry holding a subset.
//
// Regression fixture for QVAC-23763 / tetherto/qvac#4171. Before the guard, a
// card below the compiled floor enumerated, won backend selection over Vulkan,
// and then aborted at the first kernel launch instead of falling back.
//
// Skips with 77 when no CUDA registry is present, so it runs everywhere.

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

static ggml_backend_reg_t find_cuda_reg() {
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (strcmp(ggml_backend_reg_name(reg), "CUDA") == 0) {
            return reg;
        }
    }
    return nullptr;
}

// virtual devices get "-v<n>" appended to the physical card's bus id
static std::string physical_key(const char * device_id) {
    std::string id = device_id ? device_id : "";
    const size_t v = id.rfind("-v");
    if (v != std::string::npos && v + 2 < id.size()) {
        bool digits = true;
        for (size_t i = v + 2; i < id.size(); i++) {
            digits = digits && id[i] >= '0' && id[i] <= '9';
        }
        if (digits) {
            id.resize(v);
        }
    }
    return id;
}

static void print_archs(ggml_backend_reg_t reg) {
    auto get_features = (ggml_backend_get_features_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    if (get_features == nullptr) {
        printf("compiled archs: unavailable\n");
        return;
    }
    for (ggml_backend_feature * f = get_features(reg); f && f->name; f++) {
        if (strcmp(f->name, "ARCHS") == 0) {
            printf("compiled archs: %s\n", f->value);
            return;
        }
    }
    printf("compiled archs: not reported\n");
}

int main() {
    ggml_backend_load_all();

    ggml_backend_reg_t cuda = find_cuda_reg();
    if (cuda == nullptr) {
        printf("no CUDA registry, skipping\n");
        return 77;
    }

    print_archs(cuda);

    const size_t n_cuda  = ggml_backend_reg_dev_count(cuda);
    const size_t n_total = ggml_backend_dev_count();
    printf("reg=CUDA devices=%zu total_devices=%zu\n", n_cuda, n_total);

    int fails = 0;

    // The whole point of skipping at registration: a host whose every CUDA
    // device is uncovered must still have somewhere to run.
    if (n_cuda == 0 && n_total == 0) {
        fprintf(stderr, "FAIL: CUDA registered no devices and nothing else did either\n");
        fails++;
    }

    std::map<std::string, size_t> per_card;

    for (size_t i = 0; i < n_cuda; i++) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(cuda, i);

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        printf("  device %zu: name=%s id=%s desc=%s\n",
            i, props.name, props.device_id ? props.device_id : "(none)", props.description);

        per_card[physical_key(props.device_id)]++;

        // A subset registry means raw CUDA ids no longer index it. If any site
        // still resolves by index, the buffer type comes back null or bound to
        // the wrong device.
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        if (buft == nullptr) {
            fprintf(stderr, "FAIL: device %zu (%s) has a null buffer type\n", i, props.name);
            fails++;
            continue;
        }
        if (ggml_backend_buft_get_device(buft) != dev) {
            fprintf(stderr, "FAIL: device %zu (%s) buffer type is bound to a different device\n",
                i, props.name);
            fails++;
        }
    }

    // GGML_CUDA_DEVICES splits each physical card into virtual devices. They all
    // read the same properties, so they share a compute capability and must be
    // skipped or kept as a group - never split. Round-robin assignment allows a
    // difference of one between cards.
    if (!per_card.empty()) {
        size_t lo = SIZE_MAX, hi = 0;
        for (const auto & kv : per_card) {
            lo = kv.second < lo ? kv.second : lo;
            hi = kv.second > hi ? kv.second : hi;
        }
        printf("surviving cards=%zu virtual devices per card=%zu..%zu\n", per_card.size(), lo, hi);
        if (hi - lo > 1) {
            fprintf(stderr, "FAIL: one physical card was partially skipped (%zu..%zu per card)\n", lo, hi);
            fails++;
        }
    }

    if (fails > 0) {
        return 1;
    }

    printf("OK\n");
    return 0;
}
