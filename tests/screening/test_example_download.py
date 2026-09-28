"""Exercise example download, retries and preservation of local input files."""
from unittest.mock import Mock

import pytest
import requests

from screening import download


NAME = "ClinGen_region_curation_list_GRCh38.tsv"


def test_download_creates_directory_and_keeps_existing_file(tmp_path, monkeypatch, capsys):
    payload = b"synthetic download fixture\n"
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.headers = {"Content-Length": str(len(payload))}
    response.url = f"https://ftp.clinicalgenome.org/{NAME}"
    response.iter_content.return_value = [payload]
    get = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", get)
    directory = tmp_path / "new" / "raw"
    assert download.main(["--output-dir", str(directory)]) == 0
    assert (directory / NAME).read_bytes() == payload
    assert "[OK] Example download complete" in capsys.readouterr().out
    assert not (directory / (NAME + ".part")).exists()
    get.assert_called_once()
    get.reset_mock()
    assert download.main(["--output-dir", str(directory)]) == 0
    get.assert_not_called()
    assert (directory / NAME).read_bytes() == payload
    assert "kept unchanged" in capsys.readouterr().out


def test_failed_download_leaves_no_final_or_partial_file(tmp_path, monkeypatch, capsys):
    get = Mock(side_effect=requests.ConnectionError("connection unavailable"))
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    with pytest.raises(SystemExit) as error:
        download.main(["--output-dir", str(tmp_path)])
    assert error.value.code == 1
    assert get.call_count == 3
    assert not (tmp_path / NAME).exists()
    assert not (tmp_path / (NAME + ".part")).exists()
    assert "[FAIL] Example download stopped" in capsys.readouterr().err


def test_empty_existing_file_is_not_reported_as_success(tmp_path, capsys):
    (tmp_path / NAME).touch()
    with pytest.raises(SystemExit) as error:
        download.main(["--output-dir", str(tmp_path)])
    assert error.value.code == 1
    assert "not a nonempty file" in capsys.readouterr().err
