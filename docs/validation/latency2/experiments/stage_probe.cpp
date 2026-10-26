#include "preprocess.h"
#include "postprocess.h"
#include <rknn_api.h>
#include <dlfcn.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
using Clock=std::chrono::steady_clock;
template<class F> double bench(F fn,int n=30) {
    fn();auto t=Clock::now();for(int i=0;i<n;++i)fn();
    return std::chrono::duration<double,std::milli>(Clock::now()-t).count()/n;
}
int main(int argc,char**argv) {
  if(argc!=5)return 2;
  void* lib=dlopen("librknnrt.so",RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());
#define SYM(x) auto x=reinterpret_cast<decltype(&rknn_##x)>(dlsym(lib,"rknn_" #x));if(!x)throw std::runtime_error(#x)
  SYM(init);SYM(query);SYM(create_mem2);SYM(set_io_mem);SYM(run);SYM(destroy_mem);SYM(destroy);
  auto check=[](int x){if(x)throw std::runtime_error("RKNN rc="+std::to_string(x));};
  std::ifstream f(argv[1],std::ios::binary|std::ios::ate);std::vector<char> bytes(f.tellg());f.seekg(0);f.read(bytes.data(),bytes.size());
  rknn_context ctx=0;check(init(&ctx,bytes.data(),bytes.size(),0,nullptr));
  rknn_tensor_attr ia{},oa{};
  check(query(ctx,RKNN_QUERY_NATIVE_INPUT_ATTR,&ia,sizeof(ia)));
  check(query(ctx,std::string(argv[2])=="nhwc"?RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR:RKNN_QUERY_NATIVE_OUTPUT_ATTR,&oa,sizeof(oa)));
  const int h=ia.dims[1],w=ia.dims[2];
  std::cout<<"layout="<<argv[2]<<" input_bytes="<<ia.size_with_stride<<" output_bytes="<<oa.size_with_stride<<" output_fmt="<<oa.fmt<<" dims=";
  for(unsigned i=0;i<oa.n_dims;++i)std::cout<<oa.dims[i]<<",";
  std::cout<<"\n";
  auto* im=create_mem2(ctx,ia.size_with_stride,RKNN_FLAG_MEMORY_CACHEABLE);
  auto* om=create_mem2(ctx,oa.size_with_stride,RKNN_FLAG_MEMORY_CACHEABLE);
  if(!im||!om)throw std::runtime_error("alloc");
  std::memset(im->virt_addr,0,im->size);std::memset(om->virt_addr,0,om->size);
  ia.pass_through=1;oa.pass_through=1;
  check(set_io_mem(ctx,im,&ia));check(set_io_mem(ctx,om,&oa));
  aiisp::HalfTensor in{static_cast<uint16_t*>(im->virt_addr),im->size/2,size_t(ia.w_stride?ia.w_stride:w)*4,4,1,w,h};
  const size_t channels=oa.fmt==RKNN_TENSOR_NHWC?4:oa.dims[4];
  const size_t stride=oa.w_stride?oa.w_stride:(oa.fmt==RKNN_TENSOR_NHWC?oa.dims[2]:oa.dims[3]);
  aiisp::HalfTensor out{static_cast<uint16_t*>(om->virt_addr),om->size/2,stride*channels,channels,1,w,h};
  aiisp::ProcessingConfig c;c.tile=w;c.tile_height=h;c.edge_post=true;
  aiisp::Preprocessor pre(c);aiisp::Postprocessor post(c);
  std::vector<uint16_t> raw(1936*1552),result(raw.size());
  std::ifstream input(argv[3],std::ios::binary);if(!input.read(reinterpret_cast<char*>(raw.data()),raw.size()*2))throw std::runtime_error("RAW input");
  const aiisp::TileRegion region{0,0,968,776};
  std::vector<uint16_t> ci(in.elements),co(out.elements);
  auto cached_in=in;cached_in.data=ci.data();auto cached_out=out;cached_out.data=co.data();
  std::cout<<"pre_heap_ms="<<bench([&]{pre.prepare_native(raw,1936,1552,region,cached_in);})<<"\n";
  std::cout<<"pre_rknn_ms="<<bench([&]{pre.prepare_native(raw,1936,1552,region,in);})<<"\n";
  check(run(ctx,nullptr));
  std::cout<<"run_ms="<<bench([&]{check(run(ctx,nullptr));})<<"\n";
  rknn_perf_run perf{};int pr=query(ctx,RKNN_QUERY_PERF_RUN,&perf,sizeof(perf));
  std::cout<<"perf_query_rc="<<pr<<" kernel_us="<<perf.run_duration<<"\n";
  std::memcpy(co.data(),out.data,co.size()*2);
  std::cout<<"post_heap_ms="<<bench([&]{post.merge_native(cached_out,region,1936,1552,result,&raw);})<<"\n";
  std::cout<<"post_rknn_ms="<<bench([&]{post.merge_native(out,region,1936,1552,result,&raw);})<<"\n";
  c.edge_post=false;aiisp::Postprocessor plain(c);
  std::cout<<"post_no_edge_ms="<<bench([&]{plain.merge_native(out,region,1936,1552,result,&raw);})<<"\n";
  std::cout<<"memcpy_to_rknn_ms="<<bench([&]{std::memcpy(in.data,ci.data(),ci.size()*2);})<<"\n";
  post.merge_native(out,region,1936,1552,result,&raw);
  std::ofstream output(argv[4],std::ios::binary);output.write(reinterpret_cast<char*>(result.data()),result.size()*2);
  destroy_mem(ctx,im);destroy_mem(ctx,om);destroy(ctx);dlclose(lib);
}
