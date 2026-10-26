#include "raw.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

static void require(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
template<class F> static void rejects(F f) {bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,"invalid input accepted");}
int main() {
    try {
        std::vector<uint16_t> raw={0xabc,0x123,0,4095};
        std::vector<uint8_t> b(12,0x55); std::vector<uint16_t> decoded;
        aiisp::pack12(raw,2,2,b.data(),b.size(),6,aiisp::Storage::Compact12);
        require(b[0]==0xbc&&b[1]==0x3a&&b[2]==0x12,"Rockchip compact byte order wrong");
        require(b[3]==0&&b[4]==0&&b[5]==0,"row padding not cleared");
        aiisp::unpack12(b.data(),b.size(),2,2,6,aiisp::Storage::Compact12,decoded);
        require(decoded==raw,"compact roundtrip failed");
        rejects([&]{aiisp::unpack12(b.data(),11,2,2,6,aiisp::Storage::Compact12,decoded);});
        rejects([&]{aiisp::unpack12(b.data(),12,3,2,6,aiisp::Storage::Compact12,decoded);});
        const int w=270,h=262;
        raw.resize(w*h); std::mt19937 rng(623);
        for(auto& x:raw) x=uint16_t(rng()%4096);
        for(auto storage:{aiisp::Storage::Compact12,aiisp::Storage::LE16}) {
            const size_t stride=w*(storage==aiisp::Storage::Compact12?3:4)/2+16;
            b.resize(stride*h);
            aiisp::pack12(raw,w,h,b.data(),b.size(),stride,storage);
            aiisp::unpack12(b.data(),b.size(),w,h,stride,storage,decoded);
            require(decoded==raw,"padded random roundtrip failed");
        }
        std::vector<uint16_t> out;
        aiisp::tiled_filter(raw,w,h,128,[](const auto& x,auto& y){y=x;},out);
        require(out==raw,"identity changed Bayer values or missed a tile edge");
        aiisp::tiled_filter(raw,w,h,128,[](const auto& x,auto& y){aiisp::gaussian_tile(x,128,0.5f,y);},out);
        const int k[3]={1,2,1};
        for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
            float sum=0;
            for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
                const int sy=2*std::clamp(y/2+dy,0,h/2-1)+y%2;
                const int sx=2*std::clamp(x/2+dx,0,w/2-1)+x%2;
                sum+=k[dy+1]*k[dx+1]*raw[sy*w+sx]/16.0f;
            }
            const int expected=int(std::lround(0.5f*raw[y*w+x]+0.5f*sum));
            require(std::abs(int(out[y*w+x])-expected)<=1,"halo stitching disagrees with full-image oracle");
        }
        const uint16_t channel[]={100,500,2000,3900};
        for(int y=0;y<h;++y) for(int x=0;x<w;++x) raw[y*w+x]=channel[(y%2)*2+x%2];
        aiisp::tiled_filter(raw,w,h,128,[](const auto& x,auto& y){aiisp::gaussian_tile(x,128,1,y);},out);
        require(out==raw,"Bayer planes mixed or constant darkened at border");
        rejects([&]{aiisp::tiled_filter(raw,w,h,128,[](const auto&,auto& y){y.clear();},out);});
        rejects([&]{aiisp::tiled_filter(raw,w,h,128,[](const auto& x,auto& y){y=x;y[4*129]=NAN;},out);});
        std::cout<<"PASS: compact/LE16, padding, malformed frames, Bayer isolation, identity, halo seams, finite output\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
