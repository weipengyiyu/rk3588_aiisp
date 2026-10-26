#!/usr/bin/env python3
"""Build a fixed Bayer-plane filter for RK3588. No learned weights or training claim."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--tile', type=int, default=512)
    p.add_argument('--strength', type=float, default=0.5)
    p.add_argument('--onnx-only', action='store_true')
    a = p.parse_args()
    if not 3 <= a.tile <= 1024 or not 0 <= a.strength <= 1:
        p.error('invalid tile/strength')
    a.out.mkdir(parents=True, exist_ok=True)
    kernel = np.outer([1,2,1],[1,2,1]).astype(np.float32) / 16 * a.strength
    kernel[1,1] += 1-a.strength
    weights = np.repeat(kernel[None,None,:,:],4,axis=0)
    shape = [1,4,a.tile,a.tile]
    graph = helper.make_graph([
        helper.make_node('Conv',['raw','weights'],['denoised'],group=4,
                         kernel_shape=[3,3],pads=[1,1,1,1],strides=[1,1]),
    ],'fixed_bayer_gaussian',
        [helper.make_tensor_value_info('raw',TensorProto.FLOAT,shape)],
        [helper.make_tensor_value_info('denoised',TensorProto.FLOAT,shape)],
        [numpy_helper.from_array(weights,'weights')])
    model = helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],
                              producer_name='rk3588-aiisp-fixed-baseline',ir_version=8)
    onnx.checker.check_model(model)
    onnx_path = a.out / 'bayer_gaussian.onnx'
    onnx.save(model,str(onnx_path))
    manifest = dict(target='rk3588',algorithm='fixed Gaussian Bayer-plane convolution',
                    learned=False,tile=a.tile,halo=1,channels=['R','Gr','Gb','B'],
                    normalization='raw12 / 4095',strength=a.strength,
                    onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest())
    if not a.onnx_only:
        from rknn.api import RKNN
        manifest['toolkit_version']=importlib.metadata.version('rknn-toolkit2')
        rknn = RKNN(verbose=False)
        try:
            for step, fn in [
                ('config',lambda:rknn.config(target_platform='rk3588',float_dtype='float16')),
                ('load_onnx',lambda:rknn.load_onnx(model=str(onnx_path))),
                ('build',lambda:rknn.build(do_quantization=False)),
                ('export',lambda:rknn.export_rknn(str(a.out/'bayer_gaussian.rknn'))),
            ]:
                rc=fn()
                if rc not in (None,0): raise RuntimeError(f'{step} failed: {rc}')
            manifest['rknn_sha256']=hashlib.sha256((a.out/'bayer_gaussian.rknn').read_bytes()).hexdigest()
        finally:
            rknn.release()
    (a.out/'baseline.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    main()
