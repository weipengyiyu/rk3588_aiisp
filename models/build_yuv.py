#!/usr/bin/env python3
"""FP16 luma-only Gaussian example. Chroma stays in host buffers unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from rknn.api import RKNN

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--tile',type=int,default=488)
p.add_argument('--tile-height',type=int,default=392)
p.add_argument('--pack',type=int,choices=[1,4],default=4)
a=p.parse_args()
if not 3<=a.tile<=1024 or not 3<=a.tile_height<=1024:p.error('tile must be in [3,1024]')
a.out.mkdir(parents=True,exist_ok=True)
kernel=np.outer([1,2,1],[1,2,1]).astype(np.float32)/16
channels=a.pack*a.pack
k=np.zeros((channels,channels,3,3),dtype=np.float32)
for sy in range(a.pack):
    for sx in range(a.pack):
        for dy in range(-1,2):
            for dx in range(-1,2):
                iy,ix=sy+dy,sx+dx
                k[sy*a.pack+sx,(iy%a.pack)*a.pack+ix%a.pack,iy//a.pack+1,ix//a.pack+1]+=kernel[dy+1,dx+1]
shape=[1,channels,a.tile_height,a.tile]
g=helper.make_graph([helper.make_node('Conv',['luma','weights'],['denoised'],group=1,kernel_shape=[3,3],pads=[1,1,1,1])],
    'luma_gaussian',[helper.make_tensor_value_info('luma',TensorProto.FLOAT,shape)],
    [helper.make_tensor_value_info('denoised',TensorProto.FLOAT,shape)],[numpy_helper.from_array(k,'weights')])
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',13)],ir_version=8,producer_name='rk3588-aiisp-yuv-fixed')
onnx.checker.check_model(m)
base=a.out/'luma_gaussian'
onnx.save(m,str(base.with_suffix('.onnx')))
r=RKNN(verbose=False)
try:
    for name,fn in [('config',lambda:r.config(target_platform='rk3588',float_dtype='float16',optimization_level=3)),
        ('load',lambda:r.load_onnx(model=str(base.with_suffix('.onnx')))),('build',lambda:r.build(do_quantization=False)),
        ('export',lambda:r.export_rknn(str(base.with_suffix('.rknn'))))]:
        result=fn()
        if result not in (None,0):raise RuntimeError(f'{name}: {result}')
finally:r.release()
meta=dict(algorithm='gaussian-luma',learned=False,channels=channels,yuv_pack=a.pack,tile=a.tile,tile_height=a.tile_height,halo=1,
    normalization='Y / 255',chroma='preserved',range='preserve input coding; no full/limited range conversion',native_io='FP16')
for ext in ('onnx','rknn'):meta[ext+'_sha256']=hashlib.sha256(base.with_suffix('.'+ext).read_bytes()).hexdigest()
base.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
