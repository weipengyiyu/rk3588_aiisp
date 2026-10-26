#include "pipeline.h"
#include "raw.h"
#include "video.h"
#include "isp.h"
#include <utility>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unistd.h>

namespace {
volatile sig_atomic_t stopped=0;
void signal_handler(int) { stopped=1; }
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point begin) {return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();}
int integer(const std::string& text) {
    size_t used=0; const int value=std::stoi(text,&used);
    if(used!=text.size()) throw std::runtime_error("invalid integer: "+text);
    return value;
}
float number(const std::string& text) {
    size_t used=0; const float value=std::stof(text,&used);
    if(used!=text.size() || !std::isfinite(value)) throw std::runtime_error("invalid number: "+text);
    return value;
}
std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()<0 || uint64_t(f.tellg())>128*1024*1024) throw std::runtime_error("invalid input file: "+path);
    std::vector<uint8_t> bytes(static_cast<size_t>(f.tellg())); f.seekg(0);
    if(!f.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(bytes.size()))) throw std::runtime_error("input read failed");
    return bytes;
}
void write_file(const std::string& path,const void* data,size_t n) {
    if(path.empty()) return;
    std::ofstream f(path,std::ios::binary|std::ios::trunc);
    if(!f || !f.write(static_cast<const char*>(data),std::streamsize(n))) throw std::runtime_error("output write failed: "+path);
}
double save_raw(const std::string& path,const std::vector<uint16_t>& raw,int width,int height) {
    if(path.empty()) return 0;
    std::vector<uint8_t> bytes(size_t(width)*height*2);
    const auto tick=Clock::now();
    aiisp::pack12(raw,width,height,bytes.data(),bytes.size(),width*2,aiisp::Storage::LE16);
    const double pack_ms=elapsed(tick);
    write_file(path,bytes.data(),bytes.size());
    return pack_ms;
}
std::string escaped(const std::string& s) {
    std::string out;
    for(unsigned char c:s) { if(c=='"'||c=='\\') {out+='\\';out+=char(c);} else if(c<32) out+=' '; else out+=char(c); }
    return out;
}
struct Stats {
    unsigned frames=0, dropped=0, gaps=0;
    int width=1936,height=1552,halo=1;
    float black_level=0,white_level=4095;
    std::string sensor="imx623",bayer="rggb";
    std::string post_filter="none";
    int tile=512,tile_height=512;
    float edge_strength=0.75f,edge_threshold=96;
    double pre_ms=0,backend_ms=0,post_ms=0,unpack_ms=0,pack_ms=0,isp_ms=0;
    uint64_t tiles=0;
    void stages(const aiisp::StageTimings& t) {
        pre_ms+=t.pre_ms; backend_ms+=t.npu_ms; post_ms+=t.post_ms; tiles+=t.tiles;
    }
    uint64_t calls=0;
    double total_ms=0, infer_ms=0;
    double completion_window_ms=0;
    unsigned yuv_stride=0;
    bool success=false;
    bool live=false;
    std::vector<double> denoise_samples,processing_samples;
    static double percentile(std::vector<double> samples,double p) {
        if(samples.empty())return 0;
        std::sort(samples.begin(),samples.end());
        return samples[size_t(std::ceil(p*(samples.size()-1)))];
    }
    std::string error;
    void save(const std::string& path,const std::string& mode) const {
        if(path.empty()) return;
        std::ofstream f(path);
        if(!f) throw std::runtime_error("cannot write metrics: "+path);
        f<<std::fixed<<std::setprecision(3)
         <<"{\n  \"success\": "<<(success?"true":"false")<<",\n  \"mode\": \""<<escaped(mode)<<"\",\n"
         <<"  \"sensor\": \""<<escaped(sensor)<<"\", \"width\": "<<width<<", \"height\": "<<height<<",\n"
         <<"  \"bayer\": \""<<bayer<<"\", \"raw_bits\": 12, \"halo\": "<<halo<<",\n"
         <<"  \"black_level\": "<<black_level<<", \"white_level\": "<<white_level<<",\n"
         <<"  \"tile_width\": "<<tile<<", \"tile_height\": "<<tile_height<<",\n"
         <<"  \"post_filter\": \""<<post_filter<<"\", \"edge_strength\": "<<edge_strength<<", \"edge_threshold\": "<<edge_threshold<<",\n"
         <<"  \"tiles\": "<<tiles<<",\n"
         <<"  \"mean_pre_ms\": "<<(frames?pre_ms/frames:0)<<",\n"
         <<"  \"mean_backend_ms\": "<<(frames?backend_ms/frames:0)<<",\n"
         <<"  \"mean_post_ms\": "<<(frames?post_ms/frames:0)<<",\n"
         <<"  \"denoise_p50_ms\": "<<percentile(denoise_samples,0.5)<<",\n"
         <<"  \"denoise_p95_ms\": "<<percentile(denoise_samples,0.95)<<",\n"
         <<"  \"denoise_max_ms\": "<<percentile(denoise_samples,1)<<",\n"
         <<"  \"raw_to_nv12_p50_ms\": "<<percentile(processing_samples,0.5)<<",\n"
         <<"  \"raw_to_nv12_p95_ms\": "<<percentile(processing_samples,0.95)<<",\n"
         <<"  \"mean_unpack_ms\": "<<(frames?unpack_ms/frames:0)<<",\n"
         <<"  \"mean_pack_ms\": "<<(frames?pack_ms/frames:0)<<",\n"
         <<"  \"mean_isp_ms\": "<<(frames?isp_ms/frames:0)<<",\n"
         <<"  \"completed_frames\": "<<frames<<", \"npu_calls\": "<<calls<<",\n"
         <<"  \"drained_frames\": "<<dropped<<", \"sensor_sequence_gaps\": "<<gaps<<",\n"
         <<"  \"elapsed_ms\": "<<total_ms<<", \"mean_denoise_ms\": "<<(frames?infer_ms/frames:0)<<",\n"
         <<"  \"completed_fps_including_startup\": "<<(total_ms?frames*1000.0/total_ms:0)<<",\n"
         <<"  \"completed_fps_after_first\": "<<(completion_window_ms>0?(frames-1)*1000.0/completion_window_ms:0)<<",\n"
         <<"  \"execution\": \""<<(live?"live":"offline")<<"\",\n"
         <<"  \"yuv_format\": "<<(live?"\"NV12\"":"null")<<", \"yuv_stride\": "<<yuv_stride<<",\n"
         <<"  \"error\": \""<<escaped(error)<<"\"\n}\n";
        if(!f) throw std::runtime_error("metrics write failed");
    }
};
}
int main(int argc,char** argv) {
    std::map<std::string,std::string> a;
    const std::map<std::string,std::string> defaults={
        {"--width","1936"},{"--height","1552"},{"--sensor","imx623"},{"--bayer","rggb"},
        {"--halo","1"},{"--black-level","0"},{"--white-level","4095"},{"--mode","rknn"},{"--model",""},{"--tile","512"},{"--tile-height","0"},{"--strength","0.5"},
        {"--post-filter","none"},{"--edge-strength","0.75"},{"--edge-threshold","96"},
        {"--frames","30"},{"--timeout-ms","5000"},{"--startup-timeout-ms","15000"},
        {"--capture",""},{"--rawrd",""},{"--isp",""},{"--yuv",""},
        {"--storage","compact"},{"--input-raw",""},{"--input-stride","0"},
        {"--output-yuv",""},{"--dump-input",""},{"--dump-output",""},{"--json",""}};
    a=defaults;
    Stats stats; const auto start=Clock::now();
    try {
        for(int i=1;i<argc;++i) {
            std::string key=argv[i];
            if(key=="--help") {
                std::cout<<"RK3588 RAW12 -> pre -> NPU -> post -> RKISP -> NV12\n"
                         <<"Use scripts/run_pipeline.py for topology discovery and restore.\n"
                         <<"Offline: --input-raw FILE --input-stride N --storage compact|le16 --dump-output FILE\n"
                         <<"Geometry: --sensor imx623 --width 1936 --height 1552 --bayer rggb|bggr|grbg|gbrg\n"
                         <<"Model contract: --halo 1 --black-level 0 --white-level 4095\n"
                         <<"Backend: --mode rknn|rknn-native|bypass|cpu-reference --model FILE --tile 512\n"
                         <<"Fast native: --tile 976 --tile-height 784 --post-filter edge --edge-strength 0.75 --edge-threshold 96\n"
                         <<"Live: --capture NODE --rawrd NODE --isp SUBDEV --yuv NODE --frames 30\n"
                         <<"Outputs: --json FILE --output-yuv FILE --dump-input FILE --dump-output FILE\n";
                return 0;
            }
            if(!a.count(key) || i+1==argc) throw std::runtime_error("unknown/missing option: "+key);
            a[key]=argv[++i];
        }
        const int t=integer(a["--tile"]), count=integer(a["--frames"]);
        const int timeout=integer(a["--timeout-ms"]), startup=integer(a["--startup-timeout-ms"]);
        const float strength=number(a["--strength"]);
        if(t<3||t>1024||count<0||timeout<1||timeout>60000||startup<1||startup>60000||
           !std::isfinite(strength)||strength<0||strength>1)
            throw std::runtime_error("invalid numeric option");
        if((a["--mode"]=="rknn" || a["--mode"]=="rknn-native") && strength!=0.5f)
            throw std::runtime_error("--strength only controls cpu-reference; regenerate RKNN weights to change NPU strength");
        if(a["--storage"]!="compact"&&a["--storage"]!="le16") throw std::runtime_error("storage must be compact or le16");
        const auto storage=a["--storage"]=="compact"?aiisp::Storage::Compact12:aiisp::Storage::LE16;
        const int width=integer(a["--width"]),height=integer(a["--height"]);
        if(width<2 || height<2 || width>8192 || height>8192 || width%2 || height%2)
            throw std::runtime_error("width/height must be even and within [2,8192]");
        aiisp::ProcessingConfig config;
        config.tile=t; config.tile_height=integer(a["--tile-height"]);config.halo=integer(a["--halo"]);
        if(a["--post-filter"]!="none" && a["--post-filter"]!="edge")throw std::runtime_error("post-filter must be none or edge");
        config.edge_post=a["--post-filter"]=="edge";
        config.edge_strength=number(a["--edge-strength"]);config.edge_threshold=number(a["--edge-threshold"]);
        config.black_level=number(a["--black-level"]); config.white_level=number(a["--white-level"]);
        config.bayer=aiisp::parse_bayer(a["--bayer"]); config.bypass=a["--mode"]=="bypass";
        config.validate();
        if(a["--mode"]=="cpu-reference" && config.halo<1)
            throw std::runtime_error("3x3 CPU reference needs halo >= 1");
        stats.width=width; stats.height=height; stats.sensor=a["--sensor"]; stats.bayer=a["--bayer"];
        stats.halo=config.halo; stats.black_level=config.black_level; stats.white_level=config.white_level;
        stats.tile=config.tile;stats.tile_height=config.rows();stats.post_filter=a["--post-filter"];
        stats.edge_strength=config.edge_strength;stats.edge_threshold=config.edge_threshold;
        aiisp::RawPipeline pipeline(config,aiisp::make_backend(a["--mode"],a["--model"],t,strength,config.tile_height));
        std::vector<uint16_t> input,filtered;
        if(!a["--input-raw"].empty()) {
            auto bytes=read_file(a["--input-raw"]);
            const int stride=integer(a["--input-stride"]);
            if(stride<1 || a["--dump-output"].empty()) throw std::runtime_error("offline input requires stride and dump-output");
            const auto decode_tick=Clock::now();
            aiisp::unpack12(bytes.data(),bytes.size(),width,height,size_t(stride),storage,input);
            stats.unpack_ms+=elapsed(decode_tick);
            const auto tick=Clock::now();
            auto result=pipeline.process(input,width,height);
            filtered=std::move(result.raw); stats.stages(result.timings);
            stats.infer_ms=elapsed(tick);stats.denoise_samples.push_back(stats.infer_ms); stats.calls=pipeline.calls();
            save_raw(a["--dump-input"],input,width,height);
            stats.pack_ms=save_raw(a["--dump-output"],filtered,width,height); stats.frames=1;
        } else {
            stats.live=true;
            for(const char* key:{"--capture","--rawrd","--isp","--yuv"})
                if(a[key].empty()) throw std::runtime_error(std::string("live mode requires ")+key);
            // These objects outlive all frame references and stop streams on every exception.
            aiisp::VideoQueue cap(a["--capture"],false);
            cap.configure(width,height,aiisp::raw12_fourcc(config.bayer),4);
            aiisp::IspReadback isp(a["--rawrd"],a["--isp"],a["--yuv"],width,height,config.bayer);
            const auto cs=cap.format().plane_fmt[0].bytesperline;
            stats.yuv_stride=isp.stride();
            std::signal(SIGINT,signal_handler); std::signal(SIGTERM,signal_handler);
            for(unsigned i=0;i<cap.count();++i) cap.queue(i);
            cap.start();
            uint32_t previous=0;
            Clock::time_point first_completion;
            std::vector<uint8_t> last_yuv;
            aiisp::ProcessedFrame processed;
            while(!stopped && (!count || stats.frames<unsigned(count))) {
                auto frame=cap.dequeue(stats.frames?unsigned(timeout):unsigned(startup));
                // Bound draining to the queue capacity; never wait for a future Sensor frame.
                for(unsigned i=1;i<cap.count();++i) {
                    aiisp::Frame newer{};
                    if(!cap.try_dequeue(newer)) break;
                    cap.queue(frame.index); frame=newer; ++stats.dropped;
                }
                if(stats.frames) {
                    const uint32_t delta=frame.sequence-previous;
                    if(delta>1 && delta<0x80000000u) stats.gaps+=delta-1;
                }
                previous=frame.sequence;
                const auto decode_tick=Clock::now();
                aiisp::unpack12(cap.data(frame.index),frame.bytes,width,height,cs,storage,input);
                stats.unpack_ms+=elapsed(decode_tick);
                cap.queue(frame.index);
                const auto tick=Clock::now();
                aiisp::FrameMetadata metadata; metadata.sequence=frame.sequence; metadata.timestamp_ns=frame.timestamp_ns;
                pipeline.process_into(input,width,height,metadata,processed);
                stats.stages(processed.timings);
                const double denoise=elapsed(tick);stats.infer_ms+=denoise;stats.denoise_samples.push_back(denoise);stats.calls=pipeline.calls();
                auto result=isp.submit(processed,unsigned(timeout),!a["--output-yuv"].empty());
                stats.processing_samples.push_back(elapsed(decode_tick));
                stats.pack_ms+=result.pack_ms; stats.isp_ms+=result.isp_ms;
                ++stats.frames;
                if(stats.frames==1) first_completion=Clock::now();
                else stats.completion_window_ms=elapsed(first_completion);
                if(stats.frames==1 || stats.frames%10==0)
                    std::cerr<<"frame="<<stats.frames<<" sensor_seq="<<frame.sequence<<" yuv_seq="<<result.metadata.sequence
                             <<" npu_calls="<<stats.calls<<" denoise_mean_ms="<<stats.infer_ms/stats.frames<<"\n";
                // Hold only the requested last result; avoid per-frame diagnostic disk writes.
                if(!a["--output-yuv"].empty())
                    last_yuv=std::move(result.nv12);
            }
            if(stopped) throw std::runtime_error("interrupted by signal");
            isp.restore(); cap.restore();
            filtered=std::move(processed.raw);
            write_file(a["--output-yuv"],last_yuv.data(),last_yuv.size());
            save_raw(a["--dump-input"],input,width,height); save_raw(a["--dump-output"],filtered,width,height);
        }
        if((a["--mode"]=="rknn" || a["--mode"]=="rknn-native") && stats.calls==0) throw std::runtime_error("no successful NPU execution");
        stats.success=stats.frames>0; stats.total_ms=elapsed(start); stats.save(a["--json"],a["--mode"]);
        std::cerr<<"completed="<<stats.frames<<" npu_calls="<<stats.calls<<" elapsed_ms="<<stats.total_ms<<"\n";
        return stats.success?0:1;
    } catch(const std::exception& e) {
        stats.error=e.what(); stats.total_ms=elapsed(start);
        try {stats.save(a["--json"],a["--mode"]);} catch(const std::exception& m) {std::cerr<<m.what()<<"\n";}
        std::cerr<<"rk3588_aiisp: "<<e.what()<<"\n";
        return stopped?130:1;
    }
}
