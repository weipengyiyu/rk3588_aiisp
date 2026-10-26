#!/usr/bin/env python3
"""Dense diagonal Bayer kernels avoid RK3588 grouped-output channel padding.

Fixed engineering filters, not learned weights. The edge model attenuates the
Gaussian residual at high-contrast structures rather than blurring them equally.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'fast')
    p.add_argument('--tile',type=int,default=976)
    p.add_argument('--tile-height',type=int,default=784)
    p.add_argument('--algorithm',choices=['gaussian','edge'],default='edge')
    p.add_argument('--strength',type=float)
    p.add_argument('--edge-threshold',type=float,default=96.0,help='RAW12 DN at default black=0/white=4095')
    a=p.parse_args()
    strength=a.strength if a.strength is not None else (0.5 if a.algorithm=='gaussian' else 0.75)
    if not 3<=a.tile<=1024 or not 3<=a.tile_height<=1024 or not 0<=strength<=1 or not 0<a.edge_threshold<=4095:p.error('invalid filter parameters')
    a.out.mkdir(parents=True,exist_ok=True)
    k=np.outer([1,2,1],[1,2,1]).astype(np.float32)/16
    if a.algorithm=='gaussian':k*=strength;k[1,1]+=1-strength
    # Dense group=1 diagonal kernels preserve Bayer independence and let the
    # compiler use one channel block instead of one padded block per group.
    weights=np.zeros((4,4,3,3),dtype=np.float32)
    for c in range(4):weights[c,c]=k
    initial=[numpy_helper.from_array(weights,'weights')]
    nodes=[helper.make_node('Conv',['raw','weights'],['blur' if a.algorithm=='edge' else 'denoised'],
                            group=1,kernel_shape=[3,3],pads=[1,1,1,1])]
    if a.algorithm=='edge':
        for name,value in [('inverse_threshold',4095/a.edge_threshold),('one',1),('strength',strength)]:
            initial.append(numpy_helper.from_array(np.array([value],dtype=np.float32),name))
        nodes.extend([
            helper.make_node('Sub',['blur','raw'],['residual']),
            helper.make_node('Abs',['residual'],['magnitude']),
            helper.make_node('Mul',['magnitude','inverse_threshold'],['relative']),
            helper.make_node('Sub',['one','relative'],['unclamped_gate']),
            helper.make_node('Relu',['unclamped_gate'],['gate']),
            helper.make_node('Mul',['residual','gate'],['gated']),
            helper.make_node('Mul',['gated','strength'],['correction']),
            helper.make_node('Add',['raw','correction'],['denoised']),
        ])
    shape=[1,4,a.tile_height,a.tile]
    graph=helper.make_graph(nodes,'bayer_'+a.algorithm+'_dense',
        [helper.make_tensor_value_info('raw',TensorProto.FLOAT,shape)],
        [helper.make_tensor_value_info('denoised',TensorProto.FLOAT,shape)],initial)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],ir_version=8,
                            producer_name='rk3588-aiisp-fixed-fast')
    onnx.checker.check_model(model)
    onnx_path=a.out/('bayer_'+a.algorithm+'.onnx');onnx.save(model,str(onnx_path))
    from rknn.api import RKNN
    rknn=RKNN(verbose=False)
    rknn_path=onnx_path.with_suffix('.rknn')
    try:
        for name,fn in [
            ('config',lambda:rknn.config(target_platform='rk3588',float_dtype='float16',optimization_level=3)),
            ('load',lambda:rknn.load_onnx(model=str(onnx_path))),
            ('build',lambda:rknn.build(do_quantization=False)),
            ('export',lambda:rknn.export_rknn(str(rknn_path))),
        ]:
            rc=fn()
            if rc not in (None,0):raise RuntimeError(f'{name}: {rc}')
    finally:rknn.release()
    manifest=dict(algorithm=a.algorithm,learned=False,tile=a.tile,tile_height=a.tile_height,halo=1,strength=strength,
        edge_threshold_raw12=a.edge_threshold,black_level=0,white_level=4095,
        channels=['R','Gr','Gb','B'],normalization='raw12 / 4095',native_io='FP16',
        onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        rknn_sha256=hashlib.sha256(rknn_path.read_bytes()).hexdigest())
    onnx_path.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
