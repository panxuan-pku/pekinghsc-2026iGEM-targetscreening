"""Verified, retryable reference downloads shared by the screening entry point."""
from datetime import datetime, timezone
import hashlib
import time


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path, *, progress=False):
    import requests

    partial = path.with_name(path.name + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(20, 60),
                              headers={"Accept-Encoding": "identity"}) as response:
                response.raise_for_status()
                downloaded, last_update = 0, time.monotonic()
                with partial.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
                        downloaded += len(chunk)
                        if progress and time.monotonic() - last_update >= 10:
                            print(f"[INFO] {path.name}: {downloaded / 1024**2:.1f} MiB downloaded", flush=True)
                            last_update = time.monotonic()
                size = partial.stat().st_size
                expected = response.headers.get("Content-Length")
                if not size or (expected and size != int(expected)):
                    raise ValueError(f"Incomplete download: {path.name}")
                metadata = {"url": url, "resolved_url": response.url,
                            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                            "bytes": size, "sha256": sha256(partial)}
            partial.replace(path)
            return metadata
        except (requests.RequestException, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            print(f"[WARN] Retrying download {attempt + 1}/2: {path.name}", flush=True)
            time.sleep(2)
