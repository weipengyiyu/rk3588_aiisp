#include "npu.h"
#include "raw.h"
#include <cmath>
#include <rknn_api.h>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace aiisp {
class NativeBackend final : public InferenceBackend {
public:
    std::string mode;
    int tile;
    int rows;
    int channels;
    float strength;
    void* lib=nullptr;
    rknn_context ctx=0;
    uint64_t call_count=0;
    rknn_tensor_mem* input_mem=nullptr;
    rknn_tensor_mem* output_mem=nullptr;
    HalfTensor native_input{},native_output{};
    bool native_io() const override { return mode=="rknn-native"; }
    HalfTensor input_tensor() override { return native_input; }
    HalfTensor output_tensor() override { return native_output; }
    uint64_t calls() const override { return call_count; }
    decltype(&rknn_init) init=nullptr;
    decltype(&rknn_destroy) destroy=nullptr;
    decltype(&rknn_query) query=nullptr;
    decltype(&rknn_inputs_set) inputs_set=nullptr;
    decltype(&rknn_run) run_model=nullptr;
    decltype(&rknn_outputs_get) outputs_get=nullptr;
    decltype(&rknn_outputs_release) outputs_release=nullptr;
    decltype(&rknn_create_mem2) create_mem=nullptr;
    decltype(&rknn_destroy_mem) destroy_mem=nullptr;
    decltype(&rknn_set_io_mem) set_io_mem=nullptr;
    rknn_tensor_attr output_attr{};
    template<class T> void symbol(T& p,const char* name) {
        p=reinterpret_cast<T>(dlsym(lib,name));
        if(!p) throw std::runtime_error(std::string("missing RKNN symbol: ")+name);
    }
    static void check(int rc,const char* op) {
        if(rc!=RKNN_SUCC) throw std::runtime_error(std::string(op)+" rc="+std::to_string(rc));
    }
    ~NativeBackend() override {
        if(ctx && destroy_mem) {
            if(input_mem)destroy_mem(ctx,input_mem);
            if(output_mem)destroy_mem(ctx,output_mem);
        }
        if(ctx && destroy)destroy(ctx);
        if(lib)dlclose(lib);
    }
    void run_native() override {
        if(!native_io() || !input_mem || !output_mem)throw std::runtime_error("native tensors not initialized");
        // Default runtime flags flush/invalidate cached I/O around this synchronous run.
        check(run_model(ctx,nullptr),"rknn_run native");++call_count;
    }
    void allocate_native() {
        symbol(create_mem,"rknn_create_mem2");symbol(destroy_mem,"rknn_destroy_mem");
        symbol(set_io_mem,"rknn_set_io_mem");
        rknn_tensor_attr in{},out{};
        check(query(ctx,RKNN_QUERY_NATIVE_INPUT_ATTR,&in,sizeof(in)),"query native input");
        check(query(ctx,RKNN_QUERY_NATIVE_OUTPUT_ATTR,&out,sizeof(out)),"query native output");
        if(in.type!=RKNN_TENSOR_FLOAT16 || out.type!=RKNN_TENSOR_FLOAT16)
            throw std::runtime_error("native mode requires FP16 tensors");
        input_mem=create_mem(ctx,in.size_with_stride,RKNN_FLAG_MEMORY_CACHEABLE);
        output_mem=create_mem(ctx,out.size_with_stride,RKNN_FLAG_MEMORY_CACHEABLE);
        if(!input_mem || !output_mem)throw std::runtime_error("cannot allocate cached RKNN tensor memory");
        std::memset(input_mem->virt_addr,0,input_mem->size);
        std::memset(output_mem->virt_addr,0,output_mem->size);
        in.pass_through=1;out.pass_through=1;
        check(set_io_mem(ctx,input_mem,&in),"bind native input");
        check(set_io_mem(ctx,output_mem,&out),"bind native output");
        const auto view=[&](const rknn_tensor_attr& a,rknn_tensor_mem* mem) {
            HalfTensor tensor; tensor.data=static_cast<uint16_t*>(mem->virt_addr);
            tensor.elements=mem->size/2;tensor.width=tile;tensor.height=rows;tensor.channels=channels;
            if(a.fmt==RKNN_TENSOR_NHWC && a.n_dims==4 && a.dims[0]==1 &&
               a.dims[1]==unsigned(rows) && a.dims[2]==unsigned(tile) && a.dims[3]==unsigned(channels)) {
                tensor.row_stride=size_t(a.w_stride?a.w_stride:tile)*channels;
                tensor.pixel_stride=channels;tensor.channel_stride=1;
            } else if(a.fmt==RKNN_TENSOR_NC1HWC2 && a.n_dims==5 && a.dims[0]==1 &&
                      a.dims[1]>=1 && a.dims[1]<=2 && a.dims[2]>=unsigned(rows) &&
                      a.dims[3]>=unsigned(tile) && a.dims[4]>=8 && a.dims[1]*a.dims[4]>=unsigned(channels)) {
                tensor.row_stride=size_t(a.w_stride?a.w_stride:a.dims[3])*a.dims[4];
                tensor.pixel_stride=a.dims[4];tensor.channel_stride=1;tensor.channel_block=a.dims[4];
                tensor.block_stride=size_t(a.dims[2])*tensor.row_stride;
            } else throw std::runtime_error("unsupported native FP16 tensor layout or grouped output");
            tensor.validate(tile,rows,channels);
            std::cerr<<"native layout fmt="<<a.fmt<<" row_stride="<<tensor.row_stride<<" pixel_stride="<<tensor.pixel_stride<<" channel_block="<<tensor.channel_block<<" block_stride="<<tensor.block_stride<<"\n";return tensor;
        };
        native_input=view(in,input_mem);native_output=view(out,output_mem);
        std::cerr<<"native FP16 cached I/O: channels="<<channels<<" input_bytes="<<input_mem->size<<" output_bytes="<<output_mem->size<<"\n";
    }
    void load(const std::string& model) {
        std::ifstream f(model,std::ios::binary|std::ios::ate);
        if(!f || f.tellg()<=0 || uint64_t(f.tellg())>std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("cannot read RKNN model: "+model);
        std::vector<char> bytes(static_cast<size_t>(f.tellg())); f.seekg(0);
        if(!f.read(bytes.data(),std::streamsize(bytes.size()))) throw std::runtime_error("short model read");
        lib=dlopen("librknnrt.so",RTLD_NOW|RTLD_LOCAL);
        if(!lib) throw std::runtime_error(std::string("load librknnrt.so: ")+dlerror());
        symbol(init,"rknn_init"); symbol(destroy,"rknn_destroy"); symbol(query,"rknn_query");
        symbol(inputs_set,"rknn_inputs_set"); symbol(run_model,"rknn_run");
        symbol(outputs_get,"rknn_outputs_get"); symbol(outputs_release,"rknn_outputs_release");
        check(init(&ctx,bytes.data(),uint32_t(bytes.size()),0,nullptr),"rknn_init");
        rknn_input_output_num n{}; check(query(ctx,RKNN_QUERY_IN_OUT_NUM,&n,sizeof(n)),"query io");
        if(n.n_input!=1 || n.n_output!=1) throw std::runtime_error("model needs exactly one input/output");
        rknn_tensor_attr input{};
        check(query(ctx,RKNN_QUERY_INPUT_ATTR,&input,sizeof(input)),"query input");
        check(query(ctx,RKNN_QUERY_OUTPUT_ATTR,&output_attr,sizeof(output_attr)),"query output");
        for(const auto* a:{&input,&output_attr}) {
            std::cerr<<"tensor="<<a->name<<" dims="<<a->n_dims<<" [";
            for(unsigned i=0;i<a->n_dims;++i) std::cerr<<a->dims[i]<<",";
            std::cerr<<"] fmt="<<a->fmt<<" type="<<a->type<<" quant="<<a->qnt_type<<" elements="<<a->n_elems<<"\n";
            const bool nchw=a->fmt==RKNN_TENSOR_NCHW && a->n_dims==4 && a->dims[0]==1 &&
                a->dims[1]==unsigned(channels) && a->dims[2]==unsigned(rows) && a->dims[3]==unsigned(tile);
            const bool nhwc=a->fmt==RKNN_TENSOR_NHWC && a->n_dims==4 && a->dims[0]==1 &&
                a->dims[1]==unsigned(rows) && a->dims[2]==unsigned(tile) && a->dims[3]==unsigned(channels);
            // RKNN may report affine metadata even for FP16 tensors; the element type
            // determines whether this is a floating point model.
            if((!nchw && !nhwc) || a->n_elems!=unsigned(tile*rows*channels) ||
               (a->type!=RKNN_TENSOR_FLOAT16 && a->type!=RKNN_TENSOR_FLOAT32))
                throw std::runtime_error("model must be unquantized [1,C,H,W] / [1,H,W,C], same shape and configured channel count");
        }
        rknn_sdk_version v{}; check(query(ctx,RKNN_QUERY_SDK_VERSION,&v,sizeof(v)),"query version");
        std::cerr<<"RKNN api="<<v.api_version<<" driver="<<v.drv_version<<" tile="<<tile<<"\n";
        if(native_io())allocate_native();
    }
    void run(const std::vector<float>& in,std::vector<float>& out) override {
        if(in.size()!=size_t(tile)*rows*channels) throw std::runtime_error("inference input shape mismatch");
        if(mode=="cpu-reference") { gaussian_tile(in,tile,strength,out); return; }
        rknn_input x{}; x.index=0; x.buf=const_cast<float*>(in.data());
        x.size=uint32_t(in.size()*sizeof(float)); x.type=RKNN_TENSOR_FLOAT32; x.fmt=RKNN_TENSOR_NHWC;
        check(inputs_set(ctx,1,&x),"rknn_inputs_set");
        check(run_model(ctx,nullptr),"rknn_run"); ++call_count;
        rknn_output y{}; y.want_float=1;
        check(outputs_get(ctx,1,&y,nullptr),"rknn_outputs_get");
        // Release the runtime-owned allocation even if validation/conversion throws.
        try {
            if(!y.buf || y.size<in.size()*sizeof(float)) throw std::runtime_error("short RKNN result");
            const float* p=static_cast<const float*>(y.buf); out.resize(in.size());
            if(output_attr.fmt==RKNN_TENSOR_NHWC) out.assign(p,p+in.size());
            else for(int c=0;c<channels;++c) for(int j=0;j<tile*rows;++j) out[size_t(j)*channels+c]=p[size_t(c)*tile*rows+j];
        } catch(...) { outputs_release(ctx,1,&y); throw; }
        check(outputs_release(ctx,1,&y),"rknn_outputs_release");
    }
};
std::unique_ptr<InferenceBackend> make_backend(const std::string& mode,const std::string& model,int tile,float strength,int tile_height,int bayer_pack,int tensor_channels) {
    if(mode=="bypass") return nullptr;
    if(mode!="rknn" && mode!="rknn-native" && mode!="cpu-reference") throw std::runtime_error("invalid inference backend");
    const int rows=tile_height?tile_height:tile;
    if(tile<3 || tile>1024 || rows<3 || rows>1024 || !std::isfinite(strength) || strength<0 || strength>1 ||
       (tensor_channels!=0 && tensor_channels!=1 && tensor_channels!=16) || (tensor_channels!=0 && mode!="rknn-native" && mode!="rknn") || (bayer_pack!=1 && bayer_pack!=2) || (bayer_pack==2 && mode!="rknn-native") || (mode=="cpu-reference" && rows!=tile))
        throw std::runtime_error("invalid inference configuration");
    auto backend=std::make_unique<NativeBackend>();
    backend->mode=mode; backend->tile=tile; backend->rows=rows;backend->channels=tensor_channels?tensor_channels:4*bayer_pack*bayer_pack;backend->strength=strength;
    if(mode=="rknn" || mode=="rknn-native") backend->load(model);
    return backend;
}
}
