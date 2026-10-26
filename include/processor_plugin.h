#ifndef AIISP_PROCESSOR_PLUGIN_H
#define AIISP_PROCESSOR_PLUGIN_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define AIISP_PROCESSOR_ABI_V1 1u
/* NATIVE selects RKNN execution: native FP16 I/O for RAW, standard RKNN
 * I/O for YUV. It never selects a CPU fallback. */
enum { AIISP_BACKEND_NATIVE=1, AIISP_BACKEND_REFERENCE=2, AIISP_BACKEND_BYPASS=3 };
enum { AIISP_RAW_LE16=0, AIISP_RAW_COMPACT12=1 };
/* CPU-visible borrowed planes. Caller owns mapping and DMA_BUF_IOCTL_SYNC.
 * NV12/NV21 have two planes, packed YUV and RAW one. No asynchronous access,
 * allocation, C++ object, allocator or device ownership crosses this ABI. */
typedef struct { uint8_t *data; uint64_t capacity, stride; } aiisp_plane_v1;
typedef struct {
    uint32_t struct_size, fourcc, width, height, plane_count;
    uint64_t sequence, timestamp_ns;
    aiisp_plane_v1 planes[2];
} aiisp_frame_v1;
typedef struct {
    uint32_t struct_size, fourcc, width, height, backend, raw_storage;
    uint32_t tile_width, tile_height, halo, bayer_pack, cpu_threads, yuv_pack;
    float black_level, white_level, edge_strength, edge_threshold;
    uint32_t edge_post;
    const char *model; /* copied by create; native model must match its contract */
} aiisp_processor_config_v1;
typedef struct {
    uint32_t struct_size;
    uint64_t npu_calls;
    double pre_ms, backend_ms, post_ms, total_ms;
} aiisp_processor_stats_v1;
typedef struct {
    uint32_t struct_size, abi_version;
    const char *name;
    int (*create)(const aiisp_processor_config_v1 *, void **, char *, size_t);
    /* RAW12 -> same Bayer LE16; YUV8 -> same YUV format, luma-only model,
     * chroma preserved. Output is valid only on success. No in-place buffers.
     * One handle is serialized by its caller. Sequence/timestamp are preserved. */
    int (*process)(void *, const aiisp_frame_v1 *, aiisp_frame_v1 *, aiisp_processor_stats_v1 *, char *, size_t);
    void (*destroy)(void *);
} aiisp_processor_api_v1;
const aiisp_processor_api_v1 *aiisp_processor_entry(uint32_t abi_version);
#ifdef __cplusplus
}
#endif
#endif
