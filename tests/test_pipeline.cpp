#include "pipeline.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F fn) {
    bool rejected=false; try {fn();} catch(const std::exception&) {rejected=true;}
    require(rejected,"invalid input/backend output accepted");
}
class Identity final : public aiisp::InferenceBackend {
    uint64_t count_=0;
public:
    void run(const std::vector<float>& in,std::vector<float>& out) override {out=in;++count_;}
    uint64_t calls() const override {return count_;}
};
class Broken final : public aiisp::InferenceBackend {
    int mode_;
public:
    explicit Broken(int mode):mode_(mode) {}
    void run(const std::vector<float>& in,std::vector<float>& out) override {
        if(mode_==0) throw std::runtime_error("backend failed");
        out=in;
        if(mode_==1) out.pop_back();
        if(mode_==2) std::fill(out.begin(),out.end(),std::numeric_limits<float>::quiet_NaN());
    }
    uint64_t calls() const override {return 0;}
};
class NativeIdentity final : public aiisp::InferenceBackend {
    std::vector<uint16_t> input_,output_;
    int tile_,rows_;
    uint64_t count_=0;
public:
    explicit NativeIdentity(int t,int rows=0):input_(size_t(rows?rows:t)*(t+3)*4),output_(size_t(rows?rows:t)*(t+5)*8),tile_(t),rows_(rows?rows:t) {}
    bool native_io() const override { return true; }
    aiisp::HalfTensor input_tensor() override {return {input_.data(),input_.size(),size_t(tile_+3)*4,4,1,tile_,rows_};}
    aiisp::HalfTensor output_tensor() override {return {output_.data(),output_.size(),size_t(tile_+5)*8,8,1,tile_,rows_};}
    void run(const std::vector<float>&,std::vector<float>&) override {throw std::runtime_error("wrong backend entry");}
    void run_native() override {
        auto in=input_tensor(),out=output_tensor();
        for(int y=0;y<rows_;++y)for(int x=0;x<tile_;++x)for(int c=0;c<4;++c)
            out.data[out.offset(y,x,c)]=in.data[in.offset(y,x,c)];
        ++count_;
    }
    uint64_t calls() const override {return count_;}
};
}
int main() {
    try {
        const aiisp::BayerPattern patterns[]={aiisp::BayerPattern::RGGB,aiisp::BayerPattern::BGGR,
                                             aiisp::BayerPattern::GRBG,aiisp::BayerPattern::GBRG};
        const int expected[][4]={{100,200,300,400},{400,300,200,100},{200,100,400,300},{300,400,100,200}};
        // Check color meaning independently of the production mapping, not just a reversible round trip.
        for(int k=0;k<4;++k) {
            aiisp::ProcessingConfig c;c.tile=5;c.halo=2;c.bayer=patterns[k];
            aiisp::Preprocessor pre(c);std::vector<float> tile;
            pre.prepare({100,200,300,400},2,2,{0,0,1,1},tile);
            for(int channel=0;channel<4;++channel)
                require(std::abs(tile[(2*5+2)*4+channel]*4095-expected[k][channel])<0.001f,"canonical Bayer order wrong");
        }
        const int w=42,h=34;
        std::vector<uint16_t> image(w*h);std::mt19937 rng(3588);
        for(auto& value:image) value=uint16_t(rng()%4096);
        aiisp::FrameMetadata metadata;metadata.sequence=0xfffffff0;metadata.timestamp_ns=1234567890123ULL;
        metadata.exposure_us=1200;metadata.analog_gain=2;metadata.digital_gain=1.5f;
        for(auto bayer:patterns) for(int halo:{1,2}) {
            aiisp::ProcessingConfig c;c.tile=9;c.halo=halo;c.bayer=bayer;
            aiisp::RawPipeline identity(c,std::make_unique<Identity>());
            auto unchanged=identity.process(image,w,h,metadata);
            require(unchanged.raw==image,"identity changed pixels or omitted partial boundary tiles");
            require(unchanged.width==w && unchanged.height==h && unchanged.bayer==bayer,"frame geometry lost");
            require(unchanged.metadata.sequence==metadata.sequence && unchanged.metadata.timestamp_ns==metadata.timestamp_ns &&
                    unchanged.metadata.exposure_us==1200 && unchanged.metadata.analog_gain==2 && unchanged.metadata.digital_gain==1.5f,
                    "metadata changed");
            require(unchanged.timings.tiles>1 && identity.calls()==unchanged.timings.tiles,"tile count wrong");
            aiisp::RawPipeline gaussian(c,aiisp::make_backend("cpu-reference","",c.tile,0.5f));
            auto result=gaussian.process(image,w,h);
            const int kernel[]={1,2,1};
            for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
                double sum=0;
                for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
                    const int sy=2*std::clamp(y/2+dy,0,h/2-1)+y%2;
                    const int sx=2*std::clamp(x/2+dx,0,w/2-1)+x%2;
                    sum+=kernel[dy+1]*kernel[dx+1]*image[sy*w+sx]/16.0;
                }
                const int oracle=int(std::lround((image[y*w+x]+sum)*0.5));
                require(std::abs(int(result.raw[y*w+x])-oracle)<=1,"Gaussian tile seam differs from full-image oracle");
            }
            require(gaussian.calls()==0,"CPU execution incorrectly reported as NPU calls");
            aiisp::RawPipeline native(c,std::make_unique<NativeIdentity>(c.tile));
            const auto fp16=native.process(image,w,h,metadata);
            for(size_t i=0;i<image.size();++i)
                require(std::abs(int(fp16.raw[i])-int(image[i]))<=1,"native FP16 layout/stride/Bayer roundtrip wrong");
            require(native.calls()==fp16.timings.tiles,"native invocation accounting wrong");
        }
        aiisp::ProcessingConfig c;c.tile=7;c.halo=2;c.black_level=64;c.white_level=4000;
        aiisp::RawPipeline normalize(c,std::make_unique<Identity>());
        auto range=normalize.process({0,64,2000,4095},2,2).raw;
        require(range[0]==64 && range[1]==64 && range[2]==2000 && range[3]==4000,"black/white normalization wrong");
        c.bypass=true;
        aiisp::RawPipeline bypass(c,nullptr);
        require(bypass.process(image,w,h).raw==image && bypass.calls()==0,"bypass is not bit exact");
        rejects([&]{bypass.process(image,w-1,h);});
        rejects([&]{bypass.process({4096,0,0,0},2,2);});
        c.bypass=false;
        rejects([&]{aiisp::RawPipeline no_backend(c,nullptr);});
        for(int mode=0;mode<3;++mode) {
            aiisp::RawPipeline broken(c,std::make_unique<Broken>(mode));
            rejects([&]{broken.process(image,w,h);});
        }
        c.halo=4;rejects([&]{c.validate();});
        c.halo=1;c.white_level=c.black_level;rejects([&]{c.validate();});
        rejects([]{aiisp::parse_bayer("rgb");});
        require(aiisp::float_to_half(1.0f)==0x3c00 && aiisp::half_to_float(0x3800)==0.5f,"FP16 conversion wrong");
        require(std::isinf(aiisp::half_to_float(0x7c00)) && std::isnan(aiisp::half_to_float(0x7e00)),"FP16 nonfinite conversion wrong");
        c.tile=9;c.halo=1;c.black_level=0;c.white_level=4095;
        std::vector<uint16_t> invalid(9*9*8,0x7e00),dest(4);
        aiisp::HalfTensor bad{invalid.data(),invalid.size(),9*8,8,1,9,9};
        rejects([&]{aiisp::Postprocessor(c).merge_native(bad,{0,0,1,1},2,2,dest);});
        for(auto pattern:patterns) {
            c.tile_height=11;c.bayer=pattern;
            aiisp::RawPipeline rectangular(c,std::make_unique<NativeIdentity>(c.tile,c.rows()));
            aiisp::ProcessedFrame reusable;
            rectangular.process_into(image,w,h,metadata,reusable);
            const auto capacity=reusable.raw.capacity();
            rectangular.process_into(image,w,h,metadata,reusable);
            for(size_t i=0;i<image.size();++i)
                require(std::abs(int(reusable.raw[i])-int(image[i]))<=1,"rectangular native tile corrupted RAW");
            require(reusable.raw.capacity()==capacity,"output allocation was not reused");
        }
        c.bayer=aiisp::BayerPattern::RGGB;c.edge_post=true;
        std::vector<uint16_t> native(size_t(c.tile)*c.rows()*8),source(20,1000),edge(20);
        aiisp::HalfTensor view{native.data(),native.size(),size_t(c.tile)*8,8,1,c.tile,c.rows()};
        for(int x=0;x<5;++x)for(int channel=0;channel<4;++channel)
            native[view.offset(1,x+1,channel)]=aiisp::float_to_half((1000.0f+x*32)/4095);
        aiisp::Postprocessor(c).merge_native(view,{0,0,5,1},10,2,edge,&source);
        for(int x=0;x<10;++x)for(int y=0;y<2;++y) {
            const float value=aiisp::half_to_float(native[view.offset(1,x/2+1,(y%2)*2+x%2)])*4095;
            const float delta=value-1000;
            const int oracle=int(std::lround(1000+0.75f*std::max(0.0f,1-std::abs(delta)/96)*delta));
            require(std::abs(int(edge[y*10+x])-oracle)<=1,"edge gate does not match independent scalar oracle");
        }
        rejects([&]{aiisp::Postprocessor(c).merge_native(view,{0,0,5,1},10,2,edge);});
        std::cout<<"PASS: canonical Bayer order, partial tiles, halo 1/2, scalar oracle, levels, metadata, bypass and hard failures\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
}
