#!/usr/bin/env bash
# Clean only artifacts owned by this framework; independent of the caller's cwd.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
exec python3 - "$ROOT" "$@" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

parser = argparse.ArgumentParser(description='Remove RK3588 AI-ISP build, package and generated baseline artifacts.')
parser.add_argument('--dry-run', action='store_true', help='list removals without changing files')
args = parser.parse_args(sys.argv[2:])
root = Path(sys.argv[1]).resolve(strict=True)
cmake = root / 'CMakeLists.txt'
if (root == Path(root.anchor) or not cmake.is_file() or
        'project(rk3588_aiisp' not in cmake.read_text() or
        not (root / 'include/pipeline.h').is_file() or
        not (root / 'models/build_baseline.py').is_file()):
    raise SystemExit('Refusing cleanup: framework root markers do not match')

removed = 0
def remove(path):
    global removed
    # Check lexical containment first; never follow a symlink for recursive deletion.
    relative = path.relative_to(root)
    if not relative.parts or '..' in relative.parts:
        raise RuntimeError('Refusing unsafe cleanup target: ' + str(path))
    if any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
        raise RuntimeError('Refusing target under symlink: ' + str(path))
    if not path.exists() and not path.is_symlink():
        return
    print(('Would remove: ' if args.dry_run else 'Removing: ') + relative.as_posix())
    if not args.dry_run:
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)
    removed += 1

def own_cmake_build(directory):
    cache = directory / 'CMakeCache.txt'
    if directory.is_symlink() or not cache.is_file() or cache.is_symlink():
        return False
    for line in cache.read_text(errors='replace').splitlines():
        if line.startswith('CMAKE_HOME_DIRECTORY:INTERNAL='):
            return Path(line.split('=', 1)[1]).resolve() == root
    return False

# Known framework output locations, including copied executable/runtime directories.
for name in ('build', 'build-host', 'build-aarch64', 'dist', 'bin', 'lib'):
    remove(root / name)

# Also support custom CMake build directories immediately inside this project.
for child in sorted(root.iterdir()):
    if args.dry_run and child.name in ('build', 'build-host', 'build-aarch64', 'dist', 'bin', 'lib'):
        continue
    if child.is_dir() and own_cmake_build(child):
        remove(child)

# In-source CMake builds: remove generated files only after verifying ownership.
if own_cmake_build(root):
    for name in ('CMakeFiles', 'CMakeCache.txt', 'cmake_install.cmake', 'install_manifest.txt',
                 'CTestTestfile.cmake', 'DartConfiguration.tcl', 'Testing', 'compile_commands.json',
                 'rk3588_aiisp', 'test_raw', 'test_pipeline', 'libraw_core.a', 'libaiisp_core.a', 'libaiisp_v4l2.a'):
        remove(root / name)
    makefile = root / 'Makefile'
    if makefile.is_file() and not makefile.is_symlink() and 'CMAKE generated file' in makefile.read_text(errors='replace')[:512]:
        remove(makefile)
    for name in ('build.ninja', 'rules.ninja', '.ninja_deps', '.ninja_log'):
        remove(root / name)

# Remove generated baseline models only when the producer manifest and hashes match.
# A supplied/custom model is never removed just because it ends in .onnx/.rknn.
models = root / 'models'
if not models.is_symlink():
    for directory, dirs, files in os.walk(models, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        if 'baseline.json' not in files:
            continue
        manifest = Path(directory) / 'baseline.json'
        if manifest.is_symlink():
            continue
        try:
            data = json.loads(manifest.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict) or data.get('target') != 'rk3588' or data.get('learned') is not False or data.get('algorithm') != 'fixed Gaussian Bayer-plane convolution':
            continue
        complete = True
        for filename, key in (('bayer_gaussian.onnx', 'onnx_sha256'), ('bayer_gaussian.rknn', 'rknn_sha256')):
            path = manifest.parent / filename
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or not path.is_file() or not data.get(key) or hashlib.sha256(path.read_bytes()).hexdigest() != data[key]:
                print('Preserving modified/unrecognized model: ' + path.relative_to(root).as_posix())
                complete = False
                continue
            remove(path)
        if complete:
            remove(manifest)

# Python caches, not captured RAW/YUV, run directories or validation evidence.
for directory, dirs, files in os.walk(root, followlinks=False):
    for name in list(dirs):
        path = Path(directory) / name
        if path.is_symlink():
            dirs.remove(name)
        elif name == '__pycache__':
            remove(path)
            dirs.remove(name)
        elif args.dry_run and Path(directory) == root and name in ('build', 'build-host', 'build-aarch64', 'dist', 'bin', 'lib'):
            dirs.remove(name)
    for name in files:
        if name.endswith(('.pyc', '.pyo')):
            remove(Path(directory) / name)
remove(root / 'validation/framework-build.log')
print(('Dry run complete: ' if args.dry_run else 'Clean complete: ') + str(removed) + ' artifact paths')
PY
