"""The public install/test entry must fail closed, without duplicated requirements."""
import importlib.util
import json
from pathlib import Path
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "screening"


def runner():
    spec = importlib.util.spec_from_file_location("test_pipeline_runner", ROOT / "screening/check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_dependencies_include_only_core_requirements():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["project"]["dynamic"] == ["dependencies"]
    assert "dependencies" not in config["project"]
    assert "dependency-groups" not in config
    assert config["tool"]["setuptools"]["dynamic"]["dependencies"] == {"file": ["screening/requirements.txt"]}
    assert sorted(p.name for p in PIPELINE.glob("*requirements*.txt")) == [
        "requirements-cnv.txt", "requirements-dev.txt", "requirements.txt"]
    from packaging.requirements import Requirement
    groups = {name: {Requirement(line).name.lower() for line in
                    (PIPELINE / filename).read_text().splitlines()
                    if line.strip() and not line.startswith("#")}
              for name, filename in [("core", "requirements.txt"),
                                     ("cnv", "requirements-cnv.txt"),
                                     ("dev", "requirements-dev.txt")]}
    assert not groups["core"] & (groups["cnv"] | groups["dev"])
    assert {"scanpy", "anndata", "infercnvpy"} <= groups["cnv"]
    assert groups["dev"] == {"pytest"}


@pytest.mark.parametrize("name", ["test_cnv_expression_qc.py", "test_cnv_infercnv_synthetic.py"])
def test_required_dependencies_are_not_silently_skipped(name):
    assert "importorskip" not in (ROOT / "tests/screening" / name).read_text()


@pytest.mark.parametrize("kind", ["missing", "wrong_version", "broken_import"])
def test_environment_preflight_rejects_incomplete_install(tmp_path, monkeypatch, kind):
    module = runner()
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==2.4.6\n")

    def version(name):
        if kind == "missing":
            raise module.metadata.PackageNotFoundError(name)
        return "0.0.1" if kind == "wrong_version" else "2.4.6"

    def load(name):
        if kind == "broken_import":
            raise ImportError("simulated binary import failure")

    monkeypatch.setattr(module.metadata, "version", version)
    monkeypatch.setattr(module.importlib, "import_module", load)
    with pytest.raises(RuntimeError, match="numpy"):
        module.check_environment(requirements)


@pytest.mark.parametrize("body", [
    '<testsuite tests="0" failures="0" errors="0" skipped="0"/>',
    '<testsuite tests="1" failures="0" errors="0" skipped="1"/>',
    '<testsuite tests="1" failures="1" errors="0" skipped="0"/>',
])
def test_acceptance_rejects_no_tests_skips_or_failures(tmp_path, body):
    report = tmp_path / "junit.xml"
    report.write_text(body)
    with pytest.raises(RuntimeError):
        runner().check_results(report)


def test_acceptance_accepts_complete_success(tmp_path):
    report = tmp_path / "junit.xml"
    report.write_text('<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0"/></testsuites>')
    assert runner().check_results(report)["tests"] == 2


def test_runner_refuses_existing_output_without_overwriting(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    keep = output / "summary.json"
    keep.write_text("previous run")
    with pytest.raises(FileExistsError):
        runner().create_output(output)
    assert keep.read_text() == "previous run"


def test_subprocess_failure_is_not_accepted(tmp_path):
    log = tmp_path / "failed.log"
    with pytest.raises(RuntimeError, match="command exited 7"):
        runner().run_command([sys.executable, "-c", "raise SystemExit(7)"], log, {})
    assert log.is_file()


def test_failed_preflight_writes_failure_summary(tmp_path, monkeypatch):
    module = runner()

    def fail():
        raise RuntimeError("simulated incomplete install")

    monkeypatch.setattr(module, "check_environment", fail)
    monkeypatch.setenv("MPLCONFIGDIR", "previous-cache")
    monkeypatch.setenv("NUMBA_CACHE_DIR", "previous-numba-cache")
    output = tmp_path / "run"
    assert module.main(["--suite", "environment", "--output", str(output)]) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert "simulated incomplete install" in summary["error"]
    assert not summary["stages"]
    assert not (output / "pytest.log").exists()
