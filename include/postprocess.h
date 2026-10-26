#pragma once
#include "preprocess.h"
namespace aiisp {
class Postprocessor {
    ProcessingConfig config_;
    std::array<int,4> positions_;
    std::shared_ptr<RowExecutor> executor_;
public:
    explicit Postprocessor(ProcessingConfig config,std::shared_ptr<RowExecutor> executor=nullptr);
    void merge(const std::vector<float>& tensor,TileRegion region,int width,int height,
               std::vector<uint16_t>& raw) const;
    void merge_native(HalfTensor tensor,TileRegion region,int width,int height,
                      std::vector<uint16_t>& raw,const std::vector<uint16_t>* original=nullptr) const;
};
}
