#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "tensor.h"
namespace aiisp {
// Synchronous boundary: run() returns only after the output is CPU-visible.
// Implement this interface to integrate another RAW model/backend. No implicit fallback.
class InferenceBackend {
public:
    virtual ~InferenceBackend()=default;
    virtual void run(const std::vector<float>& input,std::vector<float>& output)=0;
    virtual uint64_t calls() const=0;
    virtual bool native_io() const { return false; }
    virtual HalfTensor input_tensor() { return {}; }
    virtual HalfTensor output_tensor() { return {}; }
    virtual void run_native() { throw std::runtime_error("native I/O unavailable"); }
};
std::unique_ptr<InferenceBackend> make_backend(const std::string& mode,const std::string& model,
                                             int tile,float reference_strength=0.5f,int tile_height=0,int bayer_pack=1);
}
