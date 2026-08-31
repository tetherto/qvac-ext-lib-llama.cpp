#include "llama-moe-cache.h"

#include <vector>

static void test_cold_and_hit() {
    llama_moe_cache_lru lru(2, 8, 4);
    const int32_t ids[] = { 1, 2 };
    int32_t remapped[2];
    std::vector<llama_moe_cache_lru_fill> fills;
    llama_moe_cache_lru_stats stats;

    GGML_ASSERT(lru.plan(0, ids, 2, remapped, fills, stats));
    GGML_ASSERT(stats.hits == 0);
    GGML_ASSERT(stats.misses == 2);
    GGML_ASSERT(stats.evictions == 0);
    GGML_ASSERT(fills.size() == 2);
    GGML_ASSERT(remapped[0] != remapped[1]);

    GGML_ASSERT(lru.plan(0, ids, 2, remapped, fills, stats));
    GGML_ASSERT(stats.hits == 2);
    GGML_ASSERT(stats.misses == 0);
    GGML_ASSERT(stats.evictions == 0);
    GGML_ASSERT(fills.empty());
}

static void test_duplicate_ids() {
    llama_moe_cache_lru lru(1, 8, 3);
    const int32_t ids[] = { 4, 4, 6 };
    int32_t remapped[3];
    std::vector<llama_moe_cache_lru_fill> fills;
    llama_moe_cache_lru_stats stats;

    GGML_ASSERT(lru.plan(0, ids, 3, remapped, fills, stats));
    GGML_ASSERT(stats.misses == 2);
    GGML_ASSERT(fills.size() == 2);
    GGML_ASSERT(remapped[0] == remapped[1]);
    GGML_ASSERT(remapped[0] != remapped[2]);
}

static void test_global_lru() {
    llama_moe_cache_lru lru(2, 8, 2);
    const int32_t first[] = { 1, 2 };
    const int32_t second[] = { 3 };
    const int32_t hit[] = { 2 };
    int32_t remapped[2];
    std::vector<llama_moe_cache_lru_fill> fills;
    llama_moe_cache_lru_stats stats;

    GGML_ASSERT(lru.plan(0, first, 2, remapped, fills, stats));
    GGML_ASSERT(lru.plan(1, second, 1, remapped, fills, stats));
    GGML_ASSERT(stats.misses == 1);
    GGML_ASSERT(stats.evictions == 1);

    GGML_ASSERT(lru.plan(0, hit, 1, remapped, fills, stats));
    GGML_ASSERT(stats.hits == 1);
    GGML_ASSERT(stats.misses == 0);
}

static void test_current_plan_pinning() {
    llama_moe_cache_lru lru(1, 8, 2);
    const int32_t first[] = { 0, 1 };
    const int32_t second[] = { 1, 2 };
    int32_t remapped[2];
    std::vector<llama_moe_cache_lru_fill> fills;
    llama_moe_cache_lru_stats stats;

    GGML_ASSERT(lru.plan(0, first, 2, remapped, fills, stats));
    const int32_t slot_one = remapped[1];

    GGML_ASSERT(lru.plan(0, second, 2, remapped, fills, stats));
    GGML_ASSERT(stats.hits == 1);
    GGML_ASSERT(stats.misses == 1);
    GGML_ASSERT(stats.evictions == 1);
    GGML_ASSERT(remapped[0] == slot_one);
    GGML_ASSERT(remapped[0] != remapped[1]);
}

static void test_clear_and_capacity() {
    llama_moe_cache_lru lru(1, 8, 2);
    const int32_t ids[] = { 1, 2 };
    const int32_t too_many[] = { 1, 2, 3 };
    int32_t remapped[3];
    std::vector<llama_moe_cache_lru_fill> fills;
    llama_moe_cache_lru_stats stats;

    GGML_ASSERT(lru.plan(0, ids, 2, remapped, fills, stats));
    lru.clear();
    GGML_ASSERT(lru.plan(0, ids, 2, remapped, fills, stats));
    GGML_ASSERT(stats.misses == 2);
    GGML_ASSERT(!lru.plan(0, too_many, 3, remapped, fills, stats));
}

int main() {
    test_cold_and_hit();
    test_duplicate_ids();
    test_global_lru();
    test_current_plan_pinning();
    test_clear_and_capacity();
    return 0;
}
