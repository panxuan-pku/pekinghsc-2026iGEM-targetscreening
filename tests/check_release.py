"""Check distributable files without reading ignored local data or environments."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = {".csv", ".tsv", ".h5ad", ".h5", ".npz", ".npy", ".parquet", ".pt", ".pth", ".ckpt", ".pkl", ".gz", ".zip"}


def main():
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().split("\0")
    errors = []
    for name in sorted(set(filter(None, names))):
        path = ROOT / name
        if ("workspace" in path.relative_to(ROOT).parts or path.suffix in BLOCKED
                or path.name.startswith(".env") or path.stat().st_size >= 10 * 1024**2):
            errors.append(f"Unexpected release asset: {name}")
            continue
        if path.suffix == ".py":
            try:
                compile(path.read_text(encoding="utf-8"), name, "exec")
            except SyntaxError as exc:
                errors.append(f"Invalid Python: {name}: {exc}")
        if path.suffix == ".md":
            for link in re.findall(r"\]\(([^)\s]+)\)", path.read_text(encoding="utf-8")):
                if "://" not in link and not link.startswith(("#", "mailto:")):
                    target = path.parent / link.split("#", 1)[0]
                    if not target.exists():
                        errors.append(f"Broken document link: {name}: {link}")
    for required in ("LICENSE", "README.md", "screening/environment.yml", "screening/config/screening.yaml",
                     "screening/run.py", "screening/prepare_input.py", "vct/environment.yml",
                     "vct/web/app.py", "vct/web/plotly.min.js", "docs/vct.md"):
        if required not in names:
            errors.append(f"Required file absent from release: {required}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Release check passed: {len(set(filter(None, names)))} source/document files; no bundled data or models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
