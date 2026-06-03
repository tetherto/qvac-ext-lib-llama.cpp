#include "llama-kv-cells.h"

static void test_scalar_shift_does_not_touch_ext() {
    llama_kv_cells cells;
    cells.resize(1);

    llama_kv_cell_ext ext;
    ext.x = 7;
    ext.y = 5;

    cells.pos_set(0, 10);
    cells.ext_set(0, ext);
    cells.seq_add(0, 0);

    const bool removed = cells.pos_add(0, -3);

    GGML_ASSERT(!removed);
    GGML_ASSERT(cells.pos_get(0) == 7);
    GGML_ASSERT(cells.ext_get(0).x == 7);
    GGML_ASSERT(cells.ext_get(0).y == 5);
    GGML_ASSERT(cells.get_shift(0, 0) == -3);
    GGML_ASSERT(cells.get_shift(0, 1) == 0);
    GGML_ASSERT(cells.get_shift(0, 2) == 0);
    GGML_ASSERT(cells.get_shift(0, 3) == 0);
}

static void test_mrope_shift_updates_active_axes() {
    llama_kv_cells cells;
    cells.resize(1);

    llama_kv_cell_ext ext;
    ext.x = 7;
    ext.y = 5;

    cells.pos_set(0, 10);
    cells.ext_set(0, ext);
    cells.seq_add(0, 0);

    const bool removed = cells.pos_add(0, -3, true);

    GGML_ASSERT(!removed);
    GGML_ASSERT(cells.pos_get(0) == 7);
    GGML_ASSERT(cells.ext_get(0).x == 4);
    GGML_ASSERT(cells.ext_get(0).y == 2);
    GGML_ASSERT(cells.get_shift(0, 0) == -3);
    GGML_ASSERT(cells.get_shift(0, 1) == -3);
    GGML_ASSERT(cells.get_shift(0, 2) == -3);
    GGML_ASSERT(cells.get_shift(0, 3) == 0);
}

static void test_mrope_div_tracks_axis_deltas() {
    llama_kv_cells cells;
    cells.resize(1);

    llama_kv_cell_ext ext;
    ext.x = 9;
    ext.y = 5;

    cells.pos_set(0, 11);
    cells.ext_set(0, ext);
    cells.seq_add(0, 0);

    cells.pos_div(0, 2, true);

    GGML_ASSERT(cells.pos_get(0) == 5);
    GGML_ASSERT(cells.ext_get(0).x == 4);
    GGML_ASSERT(cells.ext_get(0).y == 2);
    GGML_ASSERT(cells.get_shift(0, 0) == 6);
    GGML_ASSERT(cells.get_shift(0, 1) == 3);
    GGML_ASSERT(cells.get_shift(0, 2) == 5);
    GGML_ASSERT(cells.get_shift(0, 3) == 0);
}

static void test_mrope_negative_shift_clears_cell() {
    llama_kv_cells cells;
    cells.resize(1);

    llama_kv_cell_ext ext;
    ext.x = 3;
    ext.y = 2;

    cells.pos_set(0, 1);
    cells.ext_set(0, ext);
    cells.seq_add(0, 0);

    const bool removed = cells.pos_add(0, -2, true);

    GGML_ASSERT(removed);
    GGML_ASSERT(cells.is_empty(0));
}

int main() {
    test_scalar_shift_does_not_touch_ext();
    test_mrope_shift_updates_active_axes();
    test_mrope_div_tracks_axis_deltas();
    test_mrope_negative_shift_clears_cell();

    return 0;
}
