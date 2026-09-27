"""Centralized path resolution for VirtualCellTool.
Replaces the old hardcoded absolute paths with root-relative resolution
(from __file__), so the tool works from any location.
All imports should use these constants instead of constructing paths ad-hoc.
"""
import os

# Root = VirtualCellTool/ directory (one level above this src/ file)
VCT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORKSPACE = os.path.abspath(os.path.expanduser(os.environ.get(
    "VCT_WORKSPACE", os.path.join(os.path.dirname(VCT_ROOT), "workspace", "vct"))))

# Optional checkout override; normally SIGnature is installed by Conda/pip.
SIG_DIR = os.path.abspath(os.path.expanduser(os.environ.get(
    "SIGNATURE_DIR", os.path.join(os.path.dirname(VCT_ROOT), "workspace", "external", "SIGnature"))))

# SCimilarity model weights
MODEL_DIR = os.path.abspath(os.path.expanduser(os.environ.get(
    "VCT_MODEL_DIR", os.path.join(WORKSPACE, "models", "model_files", "scimilarity"))))

# VirtualCellTool data directory
DATA_DIR = os.path.abspath(os.path.expanduser(os.environ.get("VCT_DATA_DIR", os.path.join(WORKSPACE, "data"))))
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")

# GEARS training data directory
GEARS_DATA_DIR = os.path.join(WORKSPACE, "gears_data")

# GEARS checkpoint directory
GEARS_CKPT_DIR = os.path.join(WORKSPACE, "gears_ckpt")


def resolve(path: str) -> str:
    """Resolve a path that may be relative to the VCT workspace or absolute.
    For paths that start with ~/ or $HOME/, expand user dir.
    """
    if path.startswith("~"):
        path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.join(WORKSPACE, path)
