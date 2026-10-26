#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace aiisp {
enum class Storage { Compact12, LE16 };
// Channels are raster positions in a 2x2 Bayer cell: R, Gr, Gb, B.
void unpack12(const uint8_t* src, size_t size, int width, int height, size_t stride,
              Storage storage, std::vector<uint16_t>& dst);
void pack12(const std::vector<uint16_t>& src, int width, int height, uint8_t* dst,
            size_t size, size_t stride, Storage storage);
using TileFn = std::function<void(const std::vector<float>&, std::vector<float>&)>;
// 1 Bayer-plane pixel halo; NHWC float tiles in/out. Clamp only at image edges.
void tiled_filter(const std::vector<uint16_t>& input, int width, int height,
                  int tile_size, const TileFn& infer, std::vector<uint16_t>& output);
void gaussian_tile(const std::vector<float>& input, int tile_size, float strength,
                   std::vector<float>& output);
}
