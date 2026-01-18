# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import collect_all

def _collect(name: str):
    d, b, h = [], [], []
    try:
        d, b, h = collect_all(name)
    except Exception:
        pass
    return d, b, h

datas, binaries, hiddenimports = [], [], []
for pkg in ["torch", "av", "PyNvVideoCodec", "python_vali", "tensorrt", "tensorrt_libs"]:
    d, b, h = _collect(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["jasna/__main__.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name="jasna",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="jasna",
)

