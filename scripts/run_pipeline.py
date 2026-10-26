#!/usr/bin/env python3
"""Discover a linear RAW12 sensor, enable its manual RAW readback, run, then restore changed links/pads."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def run(*args):
    p = subprocess.run(args, text=True, capture_output=True, timeout=15,
                       env={**os.environ, 'LC_ALL':'C'})
    if p.returncode:
        raise RuntimeError(f'{args[0]} failed: {p.stderr.strip() or p.stdout.strip()}')
    return p.stdout

def parse_media(node, text):
    model = re.search(r'^model\s+(.+)$',text,re.M)
    entities = {}
    for block in re.split(r'^- entity \d+: ',text,flags=re.M)[1:]:
        name = block.split(' (',1)[0]
        n = re.search(r'device node name (\S+)',block)
        f = re.search(r'\[fmt:([^/]+)/([0-9]+)x([0-9]+)',block)
        entities[name] = dict(node=n[1] if n else '', sensor='subtype Sensor ' in block,
            code=f[1] if f else '', width=int(f[2]) if f else 0, height=int(f[3]) if f else 0,
            outputs=re.findall(r'-> "([^"]+)":\d+ \[ENABLED(?:,IMMUTABLE)?\]',block), block=block)
    return dict(node=str(node),model=model[1].strip() if model else '',entities=entities)

def reachable(g, name):
    seen=set(); todo=[name]
    while todo:
        n=todo.pop()
        if n in seen: continue
        seen.add(n)
        todo.extend(g['entities'].get(n,{}).get('outputs',[]))
    return seen

def compact_state(device):
    parent=(Path('/sys/class/video4linux')/Path(device).name/'device').resolve()
    for d in [parent,*parent.parents]:
        p=d/'compact_test'
        if not p.exists(): continue
        text=p.read_text().strip()
        if text in ('0','1'): return text=='1'
        values=text.split()
        if len(values)==4 and all(x in ('0','1') for x in values): return values[0]=='1'
        m=re.search(r'id0:\s*(not compact|compact)',text)
        if m: return m[1]=='compact'
        raise RuntimeError(f'unrecognized CIF compact state at {p}: {text}')
    raise RuntimeError('cannot determine CIF RAW packing; use --storage compact|le16 after checking the BSP')

RAW_CODES={'rggb':'SRGGB12_1X12','bggr':'SBGGR12_1X12','grbg':'SGRBG12_1X12','gbrg':'SGBRG12_1X12'}

def discover(graphs, storage='auto', sensor='imx623', width=1936, height=1552, bayer='rggb'):
    matches=[]
    for g in graphs:
        for name,s in g['entities'].items():
            if not s['sensor'] or sensor.lower() not in name.lower(): continue
            if (s['code'],s['width'],s['height'])!=(RAW_CODES[bayer],width,height):
                raise RuntimeError(f'unsupported/requested sensor mode mismatch: {s["code"]}/{s["width"]}x{s["height"]}')
            live=reachable(g,name)
            captures=[e['node'] for n,e in g['entities'].items() if n in live and n=='stream_cif_mipi_id0']
            if len(captures)!=1: continue
            for isp in graphs:
                e=isp['entities']
                if g['model'] not in e or 'rkisp_mainpath' not in reachable(isp,g['model']): continue
                if not all(n in e for n in ['rkisp_rawrd2_s','rkisp-isp-subdev','rkisp_mainpath']): continue
                if any('rkisp-isp-subdev' in e.get(n,{}).get('outputs',[]) for n in ['rkisp_rawrd0_m','rkisp_rawrd1_l','rkisp_rawrd2_s']):
                    raise RuntimeError('an ISP RAW readback link is already enabled; refuse to take ownership')
                selected=storage
                if selected=='auto': selected='compact' if compact_state(captures[0]) else 'le16'
                matches.append(dict(sensor=name,sensor_subdev=s['node'],capture_media=g['node'],
                    isp_media=isp['node'],capture=captures[0],rawrd=e['rkisp_rawrd2_s']['node'],
                    isp=e['rkisp-isp-subdev']['node'],yuv=e['rkisp_mainpath']['node'],
                    storage=selected,width=width,height=height,bayer=bayer,raw_bits=12))
    if len(matches)!=1: raise RuntimeError(f'expected exactly one complete requested RAW12 route, found {len(matches)}')
    return matches[0]

def pad_setting(media, pad):
    text=run('media-ctl','-d',media,'--get-v4l2',f'"rkisp-isp-subdev":{pad}')
    f=re.search(r'fmt:([^\s\]]+)',text)
    crop=re.search(r'(?:^|\s)crop:(\([^)]+\)/\d+x\d+)',text)
    if not f: raise RuntimeError('cannot snapshot ISP pad: '+text)
    attributes=[]
    for key in ['field','colorspace','xfer','ycbcr','quantization']:
        value=re.search(r'(?:^|\s)'+key+r':([^\s\]]+)',text)
        if value: attributes.append(f'{key}:{value[1]}')
    suffix=(' '+' '.join(attributes)) if attributes else ''
    return f'"rkisp-isp-subdev":{pad}[fmt:{f[1]}'+suffix+(f' crop:{crop[1]}' if crop else '')+']'

@contextmanager
def exclusive(media):
    path='/tmp/imx623-aiisp-'+Path(media).name+'.lock'
    fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
    try:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('another AI-ISP process owns this route')
        yield
    finally: os.close(fd)

@contextmanager
def restore_route(commands, errors):
    try:
        yield
    finally:
        for command in reversed(commands):
            try: run(*command)
            except Exception as e: errors.append(str(e))
        commands.clear()

@contextmanager
def performance_profile(enabled,errors,output_dir):
    if not enabled:
        yield
        return
    original_affinity=os.sched_getaffinity(0)
    selected=original_affinity.intersection({4,5,6,7})
    if not selected: raise RuntimeError('RK3588 performance profile needs CPU4-7')
    governors=[]
    try:
        for policy in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*')):
            related={int(x) for x in (policy/'related_cpus').read_text().split()}
            if not related.intersection(selected): continue
            path=policy/'scaling_governor'; old=path.read_text().strip()
            governors.append((path,old))
            path.write_text('performance')
        os.sched_setaffinity(0,selected)
        (output_dir/'performance-profile.json').write_text(json.dumps(dict(
            cpu_affinity=sorted(selected),governor='performance',
            original_governors={str(path):old for path,old in governors}),indent=2)+'\n')
        yield
    finally:
        for path,old in reversed(governors):
            try:path.write_text(old)
            except OSError as e:errors.append(f'CPU governor restore: {e}')
        try:os.sched_setaffinity(0,original_affinity)
        except OSError as e:errors.append(f'CPU affinity restore: {e}')

def validate_edge_model(path,tile,rows,black,white):
    meta=json.loads(path.with_suffix('.json').read_text())
    if (meta.get('algorithm'),meta.get('strength'),meta.get('tile'),meta.get('tile_height'),
        meta.get('black_level'),meta.get('white_level')) != ('gaussian',1.0,tile,rows,black,white):
        raise RuntimeError('edge post requires its pure Gaussian model and matching geometry/RAW levels')
    if meta.get('rknn_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
        raise RuntimeError('edge post model hash does not match its manifest')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sensor',default='imx623',help='sensor entity name substring; must select one route')
    p.add_argument('--width',type=int,default=1936)
    p.add_argument('--height',type=int,default=1552)
    p.add_argument('--bayer',choices=tuple(RAW_CODES),default='rggb')
    p.add_argument('--halo',type=int,default=1)
    p.add_argument('--black-level',type=float,default=0)
    p.add_argument('--white-level',type=float,default=4095)
    p.add_argument('--plan',action='store_true',help='read-only topology discovery')
    p.add_argument('--storage',choices=['auto','compact','le16'],default='auto')
    p.add_argument('--mode',choices=['bypass','rknn','rknn-native','cpu-reference'],default='rknn')
    p.add_argument('--model',type=Path,default=ROOT/'models/bayer_gaussian.rknn')
    p.add_argument('--binary',type=Path,default=ROOT/('bin/rk3588_aiisp' if (ROOT/'bin/rk3588_aiisp').is_file() else 'build-aarch64/rk3588_aiisp'))
    p.add_argument('--frames',type=int,default=30)
    p.add_argument('--output-dir',type=Path,default=ROOT/'run')
    p.add_argument('--dump-last',action='store_true')
    p.add_argument('--tile',type=int,default=512)
    p.add_argument('--tile-height',type=int,default=0,help='0=square; use 784 for the full-frame fast models')
    p.add_argument('--post-filter',choices=['none','edge'],default='none')
    p.add_argument('--edge-strength',type=float,default=0.75)
    p.add_argument('--edge-threshold',type=float,default=96)
    p.add_argument('--performance',action='store_true',help='temporarily use big CPU cores/performance governor; restored on exit')
    p.add_argument('--strength',type=float,default=0.5)
    a=p.parse_args()
    if a.frames<0: p.error('frames must be >=0')
    if not a.sensor.strip(): p.error('--sensor must not be empty')
    if a.width<2 or a.height<2 or a.width>8192 or a.height>8192 or a.width%2 or a.height%2:
        p.error('width/height must be even within [2,8192]')
    rows=a.tile_height or a.tile
    if not 3<=a.tile<=1024 or not 3<=rows<=1024 or not 0<=a.halo<=(min(a.tile,rows)-1)//2:
        p.error('invalid tile/halo')
    if not 0<=a.black_level<a.white_level<=4095: p.error('invalid RAW12 black/white levels')
    graphs=[parse_media(n,run('media-ctl','-d',str(n),'-p')) for n in sorted(Path('/dev').glob('media*'))]
    plan=discover(graphs,a.storage,a.sensor,a.width,a.height,a.bayer)
    print(json.dumps(plan,indent=2),flush=True)
    if a.plan: return 0
    if not a.binary.is_file(): raise RuntimeError('build the native binary first: '+str(a.binary))
    if a.mode in ('rknn','rknn-native') and not a.model.is_file(): raise RuntimeError('missing RKNN model: '+str(a.model))
    if a.post_filter=='edge':
        if a.mode!='rknn-native' or a.halo!=1:raise RuntimeError('edge post requires native mode and halo 1')
        validate_edge_model(a.model,a.tile,rows,a.black_level,a.white_level)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    media=plan['isp_media']; restore=[]; child=None; signalled=[False]
    def interrupted(sig,_):
        signalled[0]=True
        if child and child.poll() is None: child.send_signal(sig)
        else: raise InterruptedError('interrupted during configuration')
    previous={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    cleanup_errors=[]; rc=1
    try:
        with exclusive(media), performance_profile(a.performance,cleanup_errors,a.output_dir), restore_route(restore,cleanup_errors):
            # Check owners before changing the graph. Do not stop unrelated services.
            owners=subprocess.run(['fuser',plan['capture'],plan['rawrd'],plan['yuv'],plan['isp']],capture_output=True,text=True)
            if owners.returncode==0: raise RuntimeError('camera/ISP is in use: '+owners.stdout.strip())
            if owners.returncode!=1: raise RuntimeError('cannot verify device ownership with fuser')
            for pad,fmt in [(0,RAW_CODES[a.bayer]),(2,'YUYV8_2X8')]:
                old=pad_setting(media,pad)
                restore.append(('media-ctl','-d',media,'--set-v4l2',old))
                run('media-ctl','-d',media,'--set-v4l2',
                    f'"rkisp-isp-subdev":{pad}[fmt:{fmt}/{a.width}x{a.height} crop:(0,0)/{a.width}x{a.height}]')
            link='"rkisp_rawrd2_s":0 -> "rkisp-isp-subdev":0'
            restore.append(('media-ctl','-d',media,'--links',link+'[0]'))
            run('media-ctl','-d',media,'--links',link+'[1]')
            (a.output_dir/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
            command=[str(a.binary),'--mode',a.mode,'--model',str(a.model),'--frames',str(a.frames),
                '--tile',str(a.tile),'--strength',str(a.strength),'--storage',plan['storage'],
                '--tile-height',str(a.tile_height),'--post-filter',a.post_filter,
                '--edge-strength',str(a.edge_strength),'--edge-threshold',str(a.edge_threshold),
                '--width',str(a.width),'--height',str(a.height),'--sensor',a.sensor,'--bayer',a.bayer,
                '--halo',str(a.halo),'--black-level',str(a.black_level),'--white-level',str(a.white_level),
                '--json',str(a.output_dir/'metrics.json')]
            for key in ['capture','rawrd','isp','yuv']: command.extend(['--'+key,plan[key]])
            if a.dump_last:
                for opt,name in [('output-yuv','last.nv12'),('dump-input','input.le16.raw'),('dump-output','denoised.le16.raw')]:
                    command.extend(['--'+opt,str(a.output_dir/name)])
            env={**os.environ,'LD_LIBRARY_PATH':str(ROOT/'lib')+':'+os.environ.get('LD_LIBRARY_PATH','')}
            try:
                child=subprocess.Popen(command,env=env)
                if signalled[0]: child.send_signal(signal.SIGINT)
                rc=child.wait()
            finally:
                if child and child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)
        (a.output_dir/'cleanup.json').write_text(json.dumps(dict(restored=not cleanup_errors,errors=cleanup_errors),indent=2)+'\n')
    if cleanup_errors: raise RuntimeError('graph restoration failed: '+'; '.join(cleanup_errors))
    return 130 if signalled[0] else rc

if __name__=='__main__':
    try: raise SystemExit(main())
    except (RuntimeError,OSError,subprocess.TimeoutExpired) as e:
        print(f'run_pipeline: {e}',file=sys.stderr); raise SystemExit(1)
