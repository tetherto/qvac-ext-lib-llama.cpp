#pragma once

#include "ggml.h"
#include "mtmd.h"

#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
#include <map>
#endif

// !!! Internal header, to be used by mtmd only !!!

#define MTMD_INTERNAL_HEADER

struct clip_ctx;

struct clip_image_size {
    int width;
    int height;
};

struct clip_image_f32;
struct clip_image_u8_batch;
struct clip_image_f32_batch;

enum clip_modality {
    CLIP_MODALITY_VISION,
    CLIP_MODALITY_AUDIO,
};

enum clip_flash_attn_type {
    CLIP_FLASH_ATTN_TYPE_AUTO     = -1,
    CLIP_FLASH_ATTN_TYPE_DISABLED = 0,
    CLIP_FLASH_ATTN_TYPE_ENABLED  = 1,
};

// WARNING: value 0 is BATCHED, which is NOT the default mode (sequential is).
// A zero-initialized clip_context_params/mtmd_context_params (`{}`, memset, calloc)
// therefore selects BATCHED — the one-forward-pass path whose ne[3] batching is not
// yet verified on every backend (see the NOTE in models/qwen3vl.cpp). Do not rely on
// zero-init for the default: initialize via mtmd_context_params_default() (sets
// sequential) or set image_tile_mode explicitly. These values are part of the shipped
// API (string "0"/"1"/"2" in consumers), so they are intentionally not renumbered.
enum clip_image_tile_mode {
    CLIP_IMAGE_TILE_MODE_BATCHED    = 0, // NOT the default; zero-init lands here — see warning above
    CLIP_IMAGE_TILE_MODE_SEQUENTIAL = 1, // the default (via mtmd_context_params_default)
    CLIP_IMAGE_TILE_MODE_DISABLED   = 2,
};

struct clip_context_params {
    bool use_gpu;
    enum clip_flash_attn_type flash_attn_type;
    int image_min_tokens;
    int image_max_tokens;
    bool warmup;
    bool has_bf16_weights;
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    const char * backend_device; // optional, if null will use env var or default GPU backend
    int image_tile_mode;   // 0=batched, 1=sequential (default), 2=disabled. NOTE: 0 (batched) is the zero value but NOT the default — init via mtmd_context_params_default() or set explicitly.
    int image_max_tiles;   // override preproc_max_tiles; -1 or 0 = use GGUF/model default (only a positive value overrides)
};

struct clip_init_result {
    struct clip_ctx * ctx_v; // vision context
    struct clip_ctx * ctx_a; // audio context
};

struct clip_init_result clip_init(const char * fname, struct clip_context_params ctx_params);

void clip_free(struct clip_ctx * ctx);

size_t clip_embd_nbytes(const struct clip_ctx * ctx);
size_t clip_embd_nbytes_by_img(const struct clip_ctx * ctx, int img_w, int img_h);

int32_t clip_get_image_size (const struct clip_ctx * ctx);
int32_t clip_get_patch_size (const struct clip_ctx * ctx);
int32_t clip_get_hidden_size(const struct clip_ctx * ctx);

// TODO: should be enum, not string
const char * clip_patch_merge_type(const struct clip_ctx * ctx);

int clip_n_output_tokens(const struct clip_ctx * ctx, struct clip_image_f32 * img);

// for M-RoPE, this will be the number of token positions in X and Y directions
// for other models, X will be the total number of tokens and Y will be 1
int clip_n_output_tokens_x(const struct clip_ctx * ctx, struct clip_image_f32 * img);
int clip_n_output_tokens_y(const struct clip_ctx * ctx, struct clip_image_f32 * img);

// this should be equal to the embedding dimension of the text model
int clip_n_mmproj_embd(const struct clip_ctx * ctx);

struct clip_image_size      * clip_image_size_init(void);
struct clip_image_u8        * clip_image_u8_init (void);
struct clip_image_f32       * clip_image_f32_init(void);
struct clip_image_f32_batch * clip_image_f32_batch_init(void); // only used by libllava

// nx, ny are the output image dimensions
unsigned char * clip_image_u8_get_data(struct clip_image_u8 * img, uint32_t * nx, uint32_t * ny);

void clip_image_size_free (struct clip_image_size * img_size);
void clip_image_u8_free (struct clip_image_u8  * img);
void clip_image_f32_free(struct clip_image_f32 * img);
void clip_image_u8_batch_free (struct clip_image_u8_batch  * batch);
void clip_image_f32_batch_free(struct clip_image_f32_batch * batch);

// use for accessing underlay data of clip_image_f32_batch
size_t clip_image_f32_batch_n_images(const struct clip_image_f32_batch * batch); // equivalent to batch->size()
size_t clip_image_f32_batch_nx(const struct clip_image_f32_batch * batch, int idx); // equivalent to batch[idx]->nx
size_t clip_image_f32_batch_ny(const struct clip_image_f32_batch * batch, int idx); // equivalent to batch[idx]->ny
struct clip_image_f32 * clip_image_f32_get_img(const struct clip_image_f32_batch * batch, int idx); // equivalent to batch[idx]->data

/**
 * Build image from pixels decoded by other libraries instead of stb_image.h for better performance.
 * The memory layout is RGBRGBRGB..., input buffer length must be 3*nx*ny bytes
 */
void clip_build_img_from_pixels(const unsigned char * rgb_pixels, int nx, int ny, struct clip_image_u8 * img);

struct ggml_tensor * clip_get_newline_tensor(const struct clip_ctx * ctx);

bool clip_image_encode      (struct clip_ctx * ctx, int n_threads, struct clip_image_f32 * img, float * vec);
bool clip_image_batch_encode(struct clip_ctx * ctx, int n_threads, const struct clip_image_f32_batch * imgs, float * vec);

int clip_is_minicpmv(const struct clip_ctx * ctx);
bool clip_is_glm(const struct clip_ctx * ctx);
bool clip_is_llava(const struct clip_ctx * ctx);
// note for contributor: this clip_is_(model) pattern is deprecated
//                       do NOT add new functions like this

bool clip_encode_float_image (struct clip_ctx * ctx, int n_threads, float * img, int h, int w, float * vec);

// use by audio input
void clip_image_f32_batch_add_mel(struct clip_image_f32_batch * batch, int n_mel, int n_frames, float * mel);

bool clip_has_vision_encoder(const struct clip_ctx * ctx);
bool clip_has_audio_encoder(const struct clip_ctx * ctx);
bool clip_has_whisper_encoder(const struct clip_ctx * ctx);

#ifdef __cplusplus
// qvac: per-device memory usage of an initialised clip_ctx — weight buffers
// summed with the scheduler's compute reservations. Used by mtmd_get_memory_usage
// (and ultimately by common/fit.cpp's heuristic). Restored from upstream b9341.
std::map<ggml_backend_dev_t, size_t> clip_get_mem_usage(const struct clip_ctx * ctx);

// JSON report for the most recently completed vision graph when the active
// backend exposes structured profiling. The returned pointer is owned by ctx
// and remains valid until the next image encode or clip_free().
const char * clip_get_backend_profile_json(const struct clip_ctx * ctx);
#endif
