import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import hashlib
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('launcher',Path(__file__).resolve().parents[1]/'scripts/run_pipeline.py')
launcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

def entity(node='',sensor=False,outputs=None,code='',width=0,height=0):
    return dict(node=node,sensor=sensor,outputs=outputs or [],code=code,width=width,height=height)

def graphs():
    # Nonstandard numbering proves no /dev/videoN constant is used.
    return [dict(node='/dev/media7',model='rkcif-mipi-lvds',entities={
        'm01_back_imx623 3-001a':entity('/dev/v4l-subdev40',True,['bridge'],'SRGGB12_1X12',1936,1552),
        'bridge':entity(outputs=['stream_cif_mipi_id0']),
        'stream_cif_mipi_id0':entity('/dev/video71'),
    }),dict(node='/dev/media9',model='rkisp0',entities={
        'rkcif-mipi-lvds':entity(outputs=['rkisp-isp-subdev']),
        'rkisp-isp-subdev':entity('/dev/v4l-subdev42',outputs=['rkisp_mainpath']),
        'rkisp_mainpath':entity('/dev/video82'),
        'rkisp_rawrd2_s':entity('/dev/video87'),
    })]

class LauncherTests(unittest.TestCase):
    def test_pad_snapshot_keeps_colorimetry(self):
        text='[fmt:SRGGB12_1X12/1936x1552 field:none colorspace:raw xfer:none quantization:full-range crop:(0,0)/1936x1552]'
        with patch.object(launcher,'run',return_value=text):
            self.assertEqual(launcher.pad_setting('/dev/media9',0),'"rkisp-isp-subdev":0'+text)
    def test_performance_profile_restores_after_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);policy=root/'policy4';policy.mkdir()
            (policy/'related_cpus').write_text('4 5')
            governor=policy/'scaling_governor';governor.write_text('schedutil')
            errors=[]
            with patch.object(launcher,'Path',return_value=root),patch.object(launcher.os,'sched_getaffinity',return_value=set(range(8))),patch.object(launcher.os,'sched_setaffinity') as set_affinity:
                with self.assertRaises(ValueError):
                    with launcher.performance_profile(True,errors,root):
                        self.assertEqual(governor.read_text(),'performance')
                        raise ValueError('simulation')
            self.assertEqual(governor.read_text(),'schedutil')
            self.assertEqual(set_affinity.call_args.args,(0,set(range(8))))
            self.assertEqual(errors,[])
    def test_edge_model_contract_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            model=Path(d)/'model.rknn';model.write_bytes(b'model')
            meta=dict(algorithm='gaussian',strength=1.0,tile=976,tile_height=784,
                      black_level=0,white_level=4095,rknn_sha256=hashlib.sha256(b'model').hexdigest())
            model.with_suffix('.json').write_text(json.dumps(meta))
            launcher.validate_edge_model(model,976,784,0,4095)
            model.write_bytes(b'wrong model')
            with self.assertRaises(RuntimeError):launcher.validate_edge_model(model,976,784,0,4095)
            model.write_bytes(b'model');meta['bayer_pack']=2
            model.with_suffix('.json').write_text(json.dumps(meta))
            launcher.validate_edge_model(model,976,784,0,4095,2)
            with self.assertRaises(RuntimeError):launcher.validate_edge_model(model,976,784,0,4095,1)
    def test_imx623_route_and_arbitrary_numbering(self):
        plan=launcher.discover(graphs(),'compact')
        self.assertEqual((plan['capture'],plan['rawrd'],plan['yuv']),('/dev/video71','/dev/video87','/dev/video82'))
    def test_disconnected_or_wrong_size_rejected(self):
        g=graphs();g[1]['entities']['rkcif-mipi-lvds']['outputs']=[]
        with self.assertRaises(RuntimeError):launcher.discover(g,'compact')
        g=graphs();g[0]['entities']['m01_back_imx623 3-001a']['height']=1536
        with self.assertRaises(RuntimeError):launcher.discover(g,'compact')
    def test_existing_readback_owner_rejected(self):
        g=graphs();g[1]['entities']['rkisp_rawrd2_s']['outputs']=['rkisp-isp-subdev']
        with self.assertRaises(RuntimeError):launcher.discover(g,'compact')
    def test_configurable_sensor_size_and_bayer(self):
        for bayer,code in launcher.RAW_CODES.items():
            g=graphs();sensor=g[0]['entities'].pop('m01_back_imx623 3-001a')
            sensor.update(code=code,width=1280,height=720)
            g[0]['entities']['m02_front_testcam 4-001b']=sensor
            plan=launcher.discover(g,'le16','testcam',1280,720,bayer)
            self.assertEqual((plan['width'],plan['height'],plan['bayer']),(1280,720,bayer))
    def test_ambiguous_sensor_is_rejected(self):
        g=graphs();g[0]['entities']['m02_back_imx623 4-001a']=dict(g[0]['entities']['m01_back_imx623 3-001a'])
        with self.assertRaises(RuntimeError):launcher.discover(g,'compact')
    def test_rollback_runs_in_reverse_even_on_configuration_failure(self):
        commands=[('pad0',),('pad2',),('link',)]; errors=[]; called=[]
        with patch.object(launcher,'run',side_effect=lambda *x:called.append(x)):
            with self.assertRaises(ValueError):
                with launcher.restore_route(commands,errors):raise ValueError('configuration failed')
        self.assertEqual(called,[('link',),('pad2',),('pad0',)])
        self.assertEqual(commands,[]);self.assertEqual(errors,[])
    def test_restore_failure_is_not_hidden(self):
        errors=[]
        with patch.object(launcher,'run',side_effect=RuntimeError('busy')):
            with launcher.restore_route([('link',)],errors):pass
        self.assertEqual(errors,['busy'])

if __name__=='__main__': unittest.main()
