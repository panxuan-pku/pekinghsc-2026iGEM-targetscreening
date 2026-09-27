"""Download and verify the official SIGnature SCimilarity helper files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

import requests

from vct.src.paths import WORKSPACE

URL = "https://zenodo.org/api/records/17903196/files/model_files.tar.gz/content"
MD5 = "ec04ae13a9ecd1ebad7e78b0ea6ae39e"
SIZE = 117579614


def prepare(workspace):
    workspace = Path(workspace)
    archive = workspace / "downloads/model_files.tar.gz"
    target = workspace / "models"
    if target.exists():
        raise ValueError(f"Model directory already exists: {target}; choose a new --workspace")
    archive.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        if archive.exists() and archive.stat().st_size == SIZE:
            with archive.open("rb") as stream:
                if hashlib.file_digest(stream, "md5").hexdigest() == MD5:
                    break
        partial = archive.with_suffix(".part")
        try:
            print(f"[INFO] Downloading official model (attempt {attempt + 1}/3): {URL}", flush=True)
            with requests.get(URL, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            with partial.open("rb") as stream:
                digest = hashlib.file_digest(stream, "md5").hexdigest()
            if partial.stat().st_size != SIZE or digest != MD5:
                raise ValueError("Model size/checksum mismatch; no model published")
            partial.replace(archive)
            break
        except (requests.RequestException, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2)
    with tempfile.TemporaryDirectory(prefix=".model-", dir=workspace) as tmp:
        stage = Path(tmp)
        with tarfile.open(archive) as bundle:
            members = [member for member in bundle.getmembers()
                       if member.name.rstrip("/") == "model_files/scimilarity"
                       or member.name.startswith("model_files/scimilarity/")]
            if not members:
                raise ValueError("Official archive has no SCimilarity model")
            bundle.extractall(stage, members=members, filter="data")
        if not (stage / "model_files/scimilarity/gene_order.tsv").is_file():
            raise ValueError("Extracted model is missing gene_order.tsv")
        with archive.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        (stage / "source.json").write_text(json.dumps({"url": URL, "bytes": SIZE,
            "md5": MD5, "sha256": sha256}, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(stage), target)
    print(f"[OK] Model verified and extracted: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(WORKSPACE))
    args = parser.parse_args()
    try:
        prepare(args.workspace)
    except (OSError, ValueError, requests.RequestException, tarfile.TarError) as exc:
        parser.exit(1, f"[FAIL] {exc}\n")


if __name__ == "__main__":
    main()
