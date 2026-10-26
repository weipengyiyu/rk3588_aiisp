#!/usr/bin/env python3
"""Check the copied minimal ioctl ABI against the SDK source before compilation."""
from pathlib import Path
import re
import sys

sdk=Path(sys.argv[1])
source=(sdk/'kernel/include/uapi/linux/rk-isp2-config.h').read_text()
flat=re.sub(r'\s+',' ',source.replace('\\\n',' '))
for name,number,direction,payload in [
    ('RKISP_CMD_TRIGGER_READ_BACK',0,'W','struct isp2x_csi_trigger'),
    ('RKISP_CMD_GET_CSI_MEMORY_MODE',100,'R','int'),
    ('RKISP_CMD_SET_CSI_MEMORY_MODE',101,'W','int'),
]:
    pattern=rf'#define {name} _IO{direction}\(\s*\x27V\x27, BASE_VIDIOC_PRIVATE \+ {number}, {payload}\)'
    if not re.search(pattern,flat):raise SystemExit(f'BSP ioctl changed: {name}')
for text in ['T_START_X1 = _BITUL(4)','CSI_MEM_WORD_LITTLE_ALIGN = 1']:
    if text not in flat:raise SystemExit('BSP mode changed: '+text)
fields=re.search(r'struct isp2x_csi_trigger\s*\{(.*?)\}\s*__attribute__\s*\(\(packed\)\)',source,re.S)
if not fields:raise SystemExit('BSP packed trigger layout changed')
body=re.sub(r'/\*.*?\*/','',fields[1],flags=re.S)
if re.sub(r'\s+',' ',body).strip()!='__u64 sof_timestamp; __u64 frame_timestamp; __u32 frame_id; int times; enum isp2x_trigger_mode mode;':
    raise SystemExit('BSP trigger fields changed')
print('BSP readback ioctl ABI verified')
