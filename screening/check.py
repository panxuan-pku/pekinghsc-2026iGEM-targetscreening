#!/usr/bin/env python3
"""Portable pipeline acceptance. Run with the installed pipeline interpreter.

No installs, downloads, scientific data or existing outputs are modified.
Each invocation creates a new directory containing logs and summary.json.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "screening"
REQUIREMENTS = PIPELINE / "requirements.txt"


def check_environment(requirements=REQUIREMENTS):
    # packaging validates dependency constraints without requiring pytest.
    try:
        from packaging.requirements import Requirement
    except ImportError as exc:
        raise RuntimeError("missing packaging; reinstall screening/requirements.txt") from exc
    versions, errors = {}, []
    for line in requirements.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        req = Requirement(line)
        if req.marker and not req.marker.evaluate():
            continue
        try:
            version = metadata.version(req.name)
            versions[req.name] = version
            if version not in req.specifier:
                raise RuntimeError(f"installed {version}, required {req.specifier}")
            module = {"pyyaml": "yaml"}.get(req.name.lower(), req.name.lower().replace("-", "_"))
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"{req.name}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError(f"incomplete pipeline environment; install {requirements}:\n" + "\n".join(errors))
    return versions


def check_declarations():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    if (cfg["project"].get("dynamic") != ["dependencies"] or "dependencies" in cfg["project"]
            or "dependency-groups" in cfg or cfg.get("tool", {}).get("setuptools", {}).get(
                "dynamic", {}).get("dependencies") != {"file": ["screening/requirements.txt"]}):
        raise RuntimeError("package dependencies must read only requirements.txt")


def create_output(path=None):
    if path is not None:
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=False)
        return path
    parent = ROOT / "workspace" / "test_artifacts" / "screening"
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=parent))


def check_results(path):
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    counts = {key: sum(int(s.get(key, 0)) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    if not counts["tests"] or any(counts[k] for k in ("failures", "errors", "skipped")):
        raise RuntimeError(f"acceptance requires executed tests with no failures/errors/skips: {counts}")
    return counts


def run_command(command, log, env):
    with log.open("w") as handle:
        handle.write(f"command: {command!r}\n")
        handle.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, timeout=1800)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}; detailed log: {log}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("environment", "smoke", "core", "full"), default="full")
    parser.add_argument("--output", type=Path, help="new output directory; existing paths are refused")
    parser.add_argument("--with-cnv", action="store_true", help="also check CNV dependencies")
    parser.add_argument("--with-dev", action="store_true", help="also check test dependencies")
    args = parser.parse_args(argv)
    try:
        output = create_output(args.output)
    except OSError as exc:
        parser.error(f"cannot create a NEW test output directory: {exc}")
    summary = {"status": "running", "suite": args.suite, "started_at": datetime.now(timezone.utc).isoformat(),
               "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
               "requirements_sha256": hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest(),
               "output": str(output), "stages": {}}
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Pipeline checks: {args.suite} ===", flush=True)
    print("[INFO] Checking the environment and dependencies...", flush=True)
    # Keep plotting/JIT caches with this run, not in a user's home or source tree.
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib-cache")
    os.environ["NUMBA_CACHE_DIR"] = str(output / "numba-cache")
    env = os.environ.copy()
    # Do not let machine-local pytest options/plugins silently deselect tests.
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH"):
        env.pop(key, None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    code = 1
    try:
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("this acceptance baseline requires Python 3.11")
        check_declarations()
        summary["versions"] = check_environment()
        extras = []
        if args.with_cnv or args.suite == "full":
            extras.append("cnv")
        if args.with_dev or args.suite != "environment":
            extras.append("dev")
        summary["optional_requirements_sha256"] = {}
        for extra in extras:
            requirements = PIPELINE / f"requirements-{extra}.txt"
            summary["optional_requirements_sha256"][extra] = hashlib.sha256(requirements.read_bytes()).hexdigest()
            summary["versions"].update(check_environment(requirements))
        run_command([sys.executable, "-m", "pip", "check"], output / "pip-check.log", env)
        summary["stages"]["environment"] = "passed"
        print("[OK] Environment verified: dependency declarations, versions, imports and pip check", flush=True)
        if args.suite != "environment":
            if args.suite == "smoke":
                targets = ["tests/screening/test_environment_contract.py", "tests/screening/test_pipeline_dataflow.py",
                           "tests/screening/test_unified_screening.py"]
            elif args.suite == "core":
                # These files exercise optional single-cell code; full runs still require them.
                targets = ["tests/screening", "--ignore-glob=tests/screening/test_cnv_*.py",
                           "--ignore=tests/screening/test_optional_cli_safety.py",
                           "--ignore=tests/screening/test_sample_identity.py",
                           "--ignore=tests/screening/test_single_cell_compensation.py"]
            else:
                targets = ["tests/screening"]
            command = [sys.executable, "-m", "pytest", *targets, "-q", "--strict-config", "--strict-markers",
                       f"--junitxml={output / 'junit.xml'}", f"--basetemp={output / 'work'}"]
            print("[INFO] Running tests; detailed output is saved to pytest.log", flush=True)
            run_command(command, output / "pytest.log", env)
            summary["tests"] = check_results(output / "junit.xml")
            summary["stages"]["tests"] = "passed"
            print(f"[OK] {summary['tests']['tests']} tests passed with no failures, errors or skips", flush=True)
        summary["status"] = "passed"
        code = 0
    except (Exception, KeyboardInterrupt) as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[FAIL] {summary['error']}", file=sys.stderr, flush=True)
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(summary, indent=2))
        label = "OK" if code == 0 else "FAIL"
        print(f"\n[{label}] Checks {'passed' if code == 0 else 'failed'} ({args.suite})", flush=True)
        print(f"  Check records: {path}", flush=True)
        print("  These checks do not download reference data or produce disease screening results.", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
