"""Tests for the cross-process entropy-status file (iter-53)."""

from __future__ import annotations

import json
import os

import pytest

from qr_sampler.entropy import status_file


@pytest.fixture()
def status_path(tmp_path, monkeypatch):
    """Point the status channel at a per-test file."""
    path = tmp_path / "qr_entropy_status.json"
    monkeypatch.setenv("QR_ENTROPY_STATUS_FILE", str(path))
    return path


class TestStatusFilePath:
    def test_default_lands_in_tempdir(self, monkeypatch) -> None:
        monkeypatch.delenv("QR_ENTROPY_STATUS_FILE", raising=False)
        path = status_file.status_file_path()
        assert path is not None
        assert path.endswith("qr_entropy_status.json")

    def test_empty_env_disables(self, monkeypatch) -> None:
        monkeypatch.setenv("QR_ENTROPY_STATUS_FILE", "")
        assert status_file.status_file_path() is None

    def test_whitespace_env_disables(self, monkeypatch) -> None:
        monkeypatch.setenv("QR_ENTROPY_STATUS_FILE", "   ")
        assert status_file.status_file_path() is None


class TestWriteRead:
    def test_roundtrip(self, status_path) -> None:
        ok = status_file.write_entropy_status({"fallback_count": 3, "currently_degraded": True})
        assert ok is True
        data = status_file.read_entropy_status()
        assert data is not None
        assert data["fallback_count"] == 3
        assert data["currently_degraded"] is True

    def test_write_stamps_updated_at(self, status_path) -> None:
        status_file.write_entropy_status({})
        data = status_file.read_entropy_status()
        assert data is not None
        assert isinstance(data["updated_at"], float)

    def test_write_does_not_mutate_payload(self, status_path) -> None:
        payload: dict = {"fallback_count": 1}
        status_file.write_entropy_status(payload)
        assert "updated_at" not in payload

    def test_write_disabled_returns_false(self, monkeypatch) -> None:
        monkeypatch.setenv("QR_ENTROPY_STATUS_FILE", "")
        assert status_file.write_entropy_status({"x": 1}) is False

    def test_read_missing_returns_none(self, status_path) -> None:
        assert status_file.read_entropy_status() is None

    def test_read_corrupt_returns_none(self, status_path) -> None:
        status_path.write_text("{not json", encoding="utf-8")
        assert status_file.read_entropy_status() is None

    def test_read_non_dict_returns_none(self, status_path) -> None:
        status_path.write_text("[1, 2, 3]", encoding="utf-8")
        assert status_file.read_entropy_status() is None

    def test_read_disabled_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv("QR_ENTROPY_STATUS_FILE", "")
        assert status_file.read_entropy_status() is None

    def test_write_leaves_no_temp_file(self, status_path) -> None:
        status_file.write_entropy_status({"x": 1})
        siblings = os.listdir(status_path.parent)
        assert siblings == [status_path.name]

    def test_write_is_valid_json_on_disk(self, status_path) -> None:
        status_file.write_entropy_status({"a": "b"})
        raw = status_path.read_text(encoding="utf-8")
        assert json.loads(raw)["a"] == "b"

    def test_write_unwritable_dir_returns_false(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("QR_ENTROPY_STATUS_FILE", str(tmp_path / "no_such_dir" / "s.json"))
        assert status_file.write_entropy_status({"x": 1}) is False
