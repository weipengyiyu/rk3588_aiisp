"""Verify destructive cleanup boundaries on isolated fixtures, never the real SDK."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/clean.sh'

class CleanTests(unittest.TestCase):
    def fixture(self, base):
        root = base / 'project with spaces'
        for name in ('scripts', 'include', 'models', 'src', 'docs', 'run-recording', 'validation'):
            (root / name).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SCRIPT, root / 'scripts/clean.sh')
        (root / 'CMakeLists.txt').write_text('project(rk3588_aiisp LANGUAGES CXX)\n')
        for name in ('include/pipeline.h', 'models/build_baseline.py', 'src/main.cpp', 'docs/notes.md',
                     'models/customer.rknn', 'run-recording/input.raw', 'validation/board-results.json'):
            (root / name).write_text('keep me')
        for name in ('build-host', 'build-aarch64', 'dist', 'bin', 'lib', 'custom-output', 'unrelated-build', 'scripts/__pycache__'):
            (root / name).mkdir()
            (root / name / 'generated').write_text('artifact')
        (root / 'custom-output/CMakeCache.txt').write_text('CMAKE_HOME_DIRECTORY:INTERNAL=' + str(root) + '\n')
        (root / 'unrelated-build/CMakeCache.txt').write_text('CMAKE_HOME_DIRECTORY:INTERNAL=/another/project\n')
        (root / 'validation/framework-build.log').write_text('build output')
        manifest = dict(target='rk3588', learned=False, algorithm='fixed Gaussian Bayer-plane convolution')
        for extension in ('onnx', 'rknn'):
            payload = ('generated ' + extension).encode()
            (root / ('models/bayer_gaussian.' + extension)).write_bytes(payload)
            manifest[extension + '_sha256'] = hashlib.sha256(payload).hexdigest()
        (root / 'models/baseline.json').write_text(json.dumps(manifest))
        return root

    def run_clean(self, root, *args):
        return subprocess.run(['bash', str(root / 'scripts/clean.sh'), *args], cwd=root.parent,
                              capture_output=True, text=True)

    def test_dry_run_clean_preservation_and_idempotence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(Path(temp))
            self.assertEqual(self.run_clean(root, '--dry-run').returncode, 0)
            self.assertTrue((root / 'build-host/generated').is_file())
            result = self.run_clean(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ('build-host', 'build-aarch64', 'dist', 'bin', 'lib', 'custom-output', 'scripts/__pycache__',
                         'models/bayer_gaussian.onnx', 'models/bayer_gaussian.rknn', 'models/baseline.json', 'validation/framework-build.log'):
                self.assertFalse((root / name).exists(), name)
            for name in ('src/main.cpp', 'include/pipeline.h', 'models/build_baseline.py', 'models/customer.rknn',
                         'docs/notes.md', 'run-recording/input.raw', 'validation/board-results.json', 'unrelated-build/generated'):
                self.assertTrue((root / name).is_file(), name)
            self.assertIn('0 artifact paths', self.run_clean(root).stdout)

    def test_symlink_targets_outside_project_survive(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); root = self.fixture(base)
            outside = base / 'outside'; outside.mkdir(); (outside / 'valuable').write_text('keep')
            shutil.rmtree(root / 'build-host')
            (root / 'build-host').symlink_to(outside, target_is_directory=True)
            (root / 'models/linked').symlink_to(outside, target_is_directory=True)
            result = self.run_clean(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((outside / 'valuable').is_file())
            self.assertFalse((root / 'build-host').is_symlink())
            self.assertTrue((root / 'models/linked').is_symlink())

    def test_modified_model_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(Path(temp))
            (root / 'models/bayer_gaussian.rknn').write_text('customer replacement')
            result = self.run_clean(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / 'models/bayer_gaussian.rknn').read_text(), 'customer replacement')
            self.assertTrue((root / 'models/baseline.json').is_file())

    def test_invalid_root_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(Path(temp))
            (root / 'CMakeLists.txt').write_text('project(other)')
            result = self.run_clean(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((root / 'build-host/generated').is_file())

    def test_in_source_cmake_clean_preserves_handwritten_makefile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(Path(temp))
            for generated in ('CMakeCache.txt', 'cmake_install.cmake', 'build.ninja', 'rk3588_aiisp'):
                (root / generated).write_text('CMAKE_HOME_DIRECTORY:INTERNAL=' + str(root))
            (root / 'Makefile').write_text('all:\n\techo user makefile\n')
            result = self.run_clean(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / 'CMakeCache.txt').exists())
            self.assertFalse((root / 'rk3588_aiisp').exists())
            self.assertTrue((root / 'Makefile').is_file())

if __name__ == '__main__':
    unittest.main()
