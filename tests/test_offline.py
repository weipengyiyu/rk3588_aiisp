"""Exercise the shipped executable across file I/O, CLI and stage boundaries."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

BINARY=str(Path(sys.argv.pop(1)).resolve())

class OfflineTests(unittest.TestCase):
    def run_case(self,root,mode='bypass',extra=(),source=None):
        raw=root/'input.raw'
        values=[(i*251+23)%4096 for i in range(8*6)]
        payload=b''.join(struct.pack('<8H',*values[y*8:y*8+8])+b'padding!' for y in range(6))
        raw.write_bytes(payload if source is None else source)
        result=subprocess.run([BINARY,'--mode',mode,'--width','8','--height','6','--tile','5',
            '--input-raw',str(raw),'--input-stride','24','--storage','le16',
            '--dump-output',str(root/'out.raw'),'--json',str(root/'metrics.json'),*extra],capture_output=True,text=True)
        return result,values,json.loads((root/'metrics.json').read_text())
    def test_bit_exact_padded_input_all_bayers(self):
        for bayer in ('rggb','bggr','grbg','gbrg'):
            with self.subTest(bayer=bayer),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                result,values,metrics=self.run_case(root,extra=('--bayer',bayer))
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(list(struct.unpack('<48H',(root/'out.raw').read_bytes())),values)
                self.assertEqual(metrics['completed_frames'],1)
                self.assertEqual(metrics['npu_calls'],0)
    def test_cpu_reference_against_scalar_file_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);result,values,metrics=self.run_case(root,'cpu-reference')
            self.assertEqual(result.returncode,0,result.stderr)
            actual=struct.unpack('<48H',(root/'out.raw').read_bytes());kernel=[1,2,1]
            for y in range(6):
                for x in range(8):
                    total=0
                    for dy in (-1,0,1):
                        for dx in (-1,0,1):
                            sy=2*min(2,max(0,y//2+dy))+y%2
                            sx=2*min(3,max(0,x//2+dx))+x%2
                            total+=kernel[dy+1]*kernel[dx+1]*values[sy*8+sx]/16
                    expected=(values[y*8+x]+total)/2
                    self.assertLessEqual(abs(actual[y*8+x]-expected),1)
            self.assertEqual(metrics['tiles'],2)
            self.assertEqual(metrics['npu_calls'],0)
    def test_short_file_and_invalid_configuration_fail(self):
        for extra,source in [((),b'bad'),(('--halo','3'),None),(('--width','7'),None),
                             (('--black-level','4095'),None),(('--bayer','rgb'),None),
                             (('--tile','5junk'),None),(('--strength','nan'),None)]:
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as directory:
                result,_,metrics=self.run_case(Path(directory),extra=extra,source=source)
                self.assertNotEqual(result.returncode,0)
                self.assertFalse(metrics['success'])
    def test_missing_model_fails_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            result,_,metrics=self.run_case(Path(directory),'rknn',('--model','/missing/model.rknn'))
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(metrics['success'])
            self.assertEqual(metrics['npu_calls'],0)
            self.assertIn('model',metrics['error'])

if __name__=='__main__':unittest.main()
