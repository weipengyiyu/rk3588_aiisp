#include "processor_plugin.h"
#include <linux/videodev2.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(false)
namespace {
constexpr int w=130,h=126;
bool semi(uint32_t f){return f==V4L2_PIX_FMT_NV12 || f==V4L2_PIX_FMT_NV21;}
struct Image {
    std::vector<uint8_t> bytes;
    aiisp_frame_v1 frame{};
    explicit Image(uint32_t f,bool raw=false):bytes((w*2+8)*h*2,0xa5){
        frame.struct_size=sizeof(frame);frame.fourcc=f;frame.width=w;frame.height=h;frame.plane_count=semi(f)?2:1;
        const uint64_t stride=(semi(f)?w:w*2)+8;
        frame.planes[0]={bytes.data(),stride*h,stride};
        if(semi(f))frame.planes[1]={bytes.data()+stride*h,stride*h/2,stride};
        for(uint32_t i=0;i<frame.plane_count;++i)for(int y=0;y<(i?h/2:h);++y)for(int x=0;x<(semi(f)?w:w*2);++x)
            frame.planes[i].data[y*stride+x]=uint8_t((x*23+y*17+i*31)%256);
        for(int y=0;y<h;++y)for(int x=0;x<w;++x){
            if(raw){const uint16_t v=(x*193+y*71)%4096;frame.planes[0].data[y*stride+x*2]=v&255;frame.planes[0].data[y*stride+x*2+1]=v>>8;}
            else frame.planes[0].data[y*stride+x*(semi(f)?1:2)+(f==V4L2_PIX_FMT_UYVY?1:0)]=(x*47+y*83)%256;
        }
        frame.sequence=73;frame.timestamp_ns=123456789;
    }
};
int raw_at(const Image& f,int y,int x){const auto* p=f.frame.planes[0].data+y*f.frame.planes[0].stride+2*x;return p[0]+256*p[1];}
int luma(const Image& f,int y,int x){return f.frame.planes[0].data[y*f.frame.planes[0].stride+x*(semi(f.frame.fourcc)?1:2)+(f.frame.fourcc==V4L2_PIX_FMT_UYVY?1:0)];}
}
int main(int argc,char** argv){
    try{
        CHECK(argc==2 || argc==4 || argc==6);
        void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);CHECK(lib);
        auto entry=reinterpret_cast<const aiisp_processor_api_v1*(*)(uint32_t)>(dlsym(lib,"aiisp_processor_entry"));CHECK(entry && !entry(999));
        const auto* api=entry(1);CHECK(api && api->struct_size==sizeof(*api));
        for(bool raw:{true,false})for(uint32_t f:raw?std::vector<uint32_t>{V4L2_PIX_FMT_SRGGB12,V4L2_PIX_FMT_SBGGR12,V4L2_PIX_FMT_SGRBG12,V4L2_PIX_FMT_SGBRG12}:
            std::vector<uint32_t>{V4L2_PIX_FMT_NV12,V4L2_PIX_FMT_NV21,V4L2_PIX_FMT_YUYV,V4L2_PIX_FMT_UYVY}){
            for(uint32_t backend:{uint32_t(AIISP_BACKEND_BYPASS),uint32_t(argc>=4?AIISP_BACKEND_NATIVE:AIISP_BACKEND_REFERENCE)}){
                Image in(f,raw),out(f,raw);char error[256]{};void* handle=nullptr;
                aiisp_processor_config_v1 c{};c.struct_size=sizeof(c);c.fourcc=f;c.width=w;c.height=h;c.backend=backend;
                c.yuv_pack=1;c.tile_width=c.tile_height=8;c.halo=1;c.bayer_pack=1;c.cpu_threads=4;c.white_level=4095;c.edge_threshold=96;
                if(backend==AIISP_BACKEND_NATIVE){c.model=raw?argv[2]:argv[3];if(!raw){c.yuv_pack=4;c.tile_width=argc==6?std::atoi(argv[4]):32;c.tile_height=argc==6?std::atoi(argv[5]):32;}if(raw){c.bayer_pack=2;c.tile_width=488;c.tile_height=392;}}
                if(!api->create(&c,&handle,error,sizeof(error)))throw std::runtime_error(error);
                aiisp_processor_stats_v1 stats{};stats.struct_size=sizeof(stats);
                if(!api->process(handle,&in.frame,&out.frame,&stats,error,sizeof(error)))throw std::runtime_error(error);
                CHECK(out.frame.sequence==73 && out.frame.timestamp_ns==123456789);
                CHECK((stats.npu_calls>0)==(backend==AIISP_BACKEND_NATIVE));
                int max_error=0;
                for(int y=0;y<h;++y)for(int x=0;x<w;++x){
                    int sum=0;
                    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
                        const int yy=raw?2*std::clamp(y/2+dy,0,h/2-1)+y%2:std::clamp(y+dy,0,h-1);
                        const int xx=raw?2*std::clamp(x/2+dx,0,w/2-1)+x%2:std::clamp(x+dx,0,w-1);
                        sum+=(raw?raw_at(in,yy,xx):luma(in,yy,xx))*(dy?1:2)*(dx?1:2);
                    }
                    const int expected=backend==AIISP_BACKEND_BYPASS?(raw?raw_at(in,y,x):luma(in,y,x)):(sum+8)/16;
                    max_error=std::max(max_error,std::abs((raw?raw_at(out,y,x):luma(out,y,x))-expected));
                }
                std::cout<<"oracle max_error="<<max_error<<" format="<<f<<" backend="<<backend<<"\n";
                CHECK(max_error<=(backend==AIISP_BACKEND_BYPASS?0:raw?2:1));
                if(!raw){
                    if(semi(f))for(int y=0;y<h/2;++y)CHECK(std::equal(in.frame.planes[1].data+y*in.frame.planes[1].stride,in.frame.planes[1].data+y*in.frame.planes[1].stride+w,out.frame.planes[1].data+y*out.frame.planes[1].stride));
                    else for(int y=0;y<h;++y)for(int x=0;x<w;++x){const int offset=x*2+(f==V4L2_PIX_FMT_UYVY?0:1);CHECK(in.frame.planes[0].data[y*in.frame.planes[0].stride+offset]==out.frame.planes[0].data[y*out.frame.planes[0].stride+offset]);}
                }
                for(uint32_t i=0;i<out.frame.plane_count;++i)for(int y=0;y<(i?h/2:h);++y)
                    for(size_t x=semi(f)?w:w*2;x<out.frame.planes[i].stride;++x)CHECK(out.frame.planes[i].data[y*out.frame.planes[i].stride+x]==0);
                auto bad=in.frame;bad.planes[0].capacity=1;CHECK(!api->process(handle,&bad,&out.frame,&stats,error,sizeof(error)));
                CHECK(!api->process(handle,&in.frame,&in.frame,&stats,error,sizeof(error)));
                if(semi(f)){auto overlap=out.frame;overlap.planes[1].data=overlap.planes[0].data;CHECK(!api->process(handle,&in.frame,&overlap,&stats,error,sizeof(error)));}
                CHECK(api->process(handle,&in.frame,&out.frame,&stats,error,sizeof(error))); // handle reuse after rejected frame
                api->destroy(handle);
                std::cout<<"format="<<f<<" backend="<<backend<<" max_error="<<max_error<<" npu_calls="<<stats.npu_calls<<" total_ms="<<stats.total_ms<<"\n";
            }
        }
        aiisp_processor_config_v1 bad{};void* handle=reinterpret_cast<void*>(1);char error[32];
        CHECK(!api->create(&bad,&handle,error,sizeof(error)) && !handle);
        dlclose(lib);std::cout<<"PASS: plugin ABI, RAW/YUV oracle, chroma, strides, partial tiles, rejection/reuse\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
