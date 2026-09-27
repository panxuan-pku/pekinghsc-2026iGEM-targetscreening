"""Resolve relocated VCT and external SIGnature without machine-local paths."""
import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

ROOT = (Path(__file__).resolve().parents[2] / "vct")


class RelocatedPathsTests(unittest.TestCase):
    def test_default_and_override(self):
        with patch.dict(os.environ, {}, clear=True):
            paths = runpy.run_path(str(ROOT / 'src/paths.py'))
        self.assertEqual(Path(paths['VCT_ROOT']), ROOT)
        self.assertEqual(Path(paths['SIG_DIR']), ROOT.parent / 'workspace/external/SIGnature')
        self.assertEqual(Path(paths['DATA_DIR']), ROOT.parent / 'workspace/vct/data')
        with patch.dict(os.environ, {'VCT_MODEL_DIR': str(ROOT / 'external-model')}):
            paths = runpy.run_path(str(ROOT / 'src/paths.py'))
        self.assertEqual(Path(paths['MODEL_DIR']), ROOT / 'external-model')

    def test_launcher_uses_active_python_and_portable_browser(self):
        launcher = (ROOT / 'src/web_launcher.py').read_text()
        self.assertIn('sys.executable', launcher)
        self.assertIn('webbrowser.open', launcher)
        self.assertNotIn('.venv', launcher)
