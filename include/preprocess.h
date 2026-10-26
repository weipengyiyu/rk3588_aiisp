#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "tensor.h"
#include "parallel.h"

namespace aiisp {
enum class BayerPattern { RGGB, BGGR, GRBG, GBRG };
BayerPattern parse_bayer(const std::string& name);
const char* bayer_name(BayerPattern pattern);
// Canonical channels: R, Gr (green on red row), Gb, B.
std::array<int,4> bayer_positions(BayerPattern pattern);
struct ProcessingConfig {
    int tile=512;
    int tile_height=0; // 0 means square; rectangular native models avoid unused rows.
    int bayer_pack=1; // Bayer-cell block side: 1 -> 4 channels, 2 -> 16 channels.
    int cpu_threads=1;
    int halo=1; // In half-resolution Bayer-plane pixels; set for the model receptive field.
    float black_level=0.0f;
    float white_level=4095.0f;
    BayerPattern bayer=BayerPattern::RGGB;
    bool bypass=false;
    bool edge_post=false;
    float edge_strength=0.75f,edge_threshold=96.0f;
    int rows() const { return tile_height?tile_height:tile; }
    int channels() const { return 4*bayer_pack*bayer_pack; }
    void validate() const;
};
struct TileRegion { int x, y, width, height; };
class Preprocessor {
    ProcessingConfig config_;
    std::array<int,4> positions_;
    std::shared_ptr<RowExecutor> executor_;
public:
    explicit Preprocessor(ProcessingConfig config,std::shared_ptr<RowExecutor> executor=nullptr);
    void prepare(const std::vector<uint16_t>& raw, int width, int height,
                 TileRegion region, std::vector<float>& tensor) const;
    void prepare_native(const std::vector<uint16_t>& raw,int width,int height,
                        TileRegion region,HalfTensor tensor) const;
};
}
