# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Trickcal Bingo AI Assistant.
Builds a standalone executable with all dependencies bundled.
"""

import sys
from pathlib import Path

block_cipher = None

# Determine the base path
BASE_PATH = Path(SPECPATH)

# Data files to include
datas = [
    # Model files (required for inference)
    (str(BASE_PATH / 'model' / 'best_model.pt'), 'model'),
    (str(BASE_PATH / 'model' / 'best_model_no_store.pt'), 'model'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'numpy',
    'PySide6',
    'PySide6.QtWidgets',
    'PySide6.QtGui',
    'PySide6.QtCore',
    'gymnasium',
    'gymnasium.spaces',
]

# Exclude unnecessary modules to reduce size
excludes = [
    'wandb',
    'tensorboard',
    'stable_baselines3',
    'sb3_contrib',
    'matplotlib',
    'PIL',
    'tkinter',
    # 'unittest',  # PyTorch uses unittest.mock internally
    'test',
    'tests',
]

a = Analysis(
    ['main.py'],
    pathex=[str(BASE_PATH)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out unnecessary torch/CUDA files to reduce size (optional)
# Uncomment if you want CPU-only build
# a.binaries = [x for x in a.binaries if not x[0].startswith('torch/lib/cudnn')]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TrickcalBingo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one: icon='icon.ico'
)

# Mac-specific: Create .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='TrickcalBingo.app',
        icon=None,  # Add icon path here if you have one: icon='icon.icns'
        bundle_identifier='com.trickcal.bingo',
        info_plist={
            'CFBundleName': 'Trickcal Bingo',
            'CFBundleDisplayName': 'Trickcal Bingo AI Assistant',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
        },
    )
