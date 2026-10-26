#!/usr/bin/env python3
"""Compare completed NPU RAW12 output with CPU full-pipeline reference."""
import argparse
import array
import json
from pathlib import Path
import sys

def read(path,width,height):
    b=path.read_bytes()
    if len(b)!=width*height*2:raise ValueError(f'wrong RAW length: {path}')
    x=array.array('H');x.frombytes(b)
    if sys.byteorder!='little':x.byteswap()
    if max(x)>4095:raise ValueError('RAW value exceeds 12 bits')
    return x

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--max-error',type=int,default=2)
    p.add_argument('--width',type=int,default=1936)
    p.add_argument('--height',type=int,default=1552)
    p.add_argument('--json',type=Path)
    a=p.parse_args()
    if a.width<2 or a.height<2 or a.width%2 or a.height%2 or a.max_error<0:p.error('invalid geometry/error threshold')
    src,out,ref=[read(x,a.width,a.height) for x in [a.input,a.output,a.reference]]
    maximum=total=changed=0
    for s,o,r in zip(src,out,ref):
        delta=abs(o-r);maximum=max(maximum,delta);total+=delta;changed+=s!=o
    result=dict(pixels=len(out),max_absolute_error_raw12=maximum,
                mean_absolute_error_raw12=total/len(out),pixels_changed=changed,
                passed=maximum<=a.max_error)
    print(json.dumps(result,indent=2))
    if a.json:a.json.write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
