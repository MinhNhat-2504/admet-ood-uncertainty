"""The lock must refuse to relock, and must detect any change to a frozen input."""

import json
import shutil

import pytest

from protocol import lock as lock_module


@pytest.fixture
def project(tmp_path):
    """A miniature project tree with the same layout the real lock expects."""
    for name in lock_module.PROTOCOL_FILES + lock_module.DATA_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"content of {name}\n", encoding="utf-8")
    source = tmp_path / "src" / "admetood"
    source.mkdir(parents=True)
    (source / "model.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def test_create_then_verify_round_trips(project):
    path = project / "protocol" / "LOCK.json"
    created = lock_module.create(project, path)
    assert created["status"] == lock_module.STATUS
    assert "data/raw/expansionrx/expansion_data_test.csv" in created["digests"]
    assert lock_module.verify(project, path)["created_utc"] == created["created_utc"]


def test_relocking_is_refused(project):
    path = project / "protocol" / "LOCK.json"
    lock_module.create(project, path)
    with pytest.raises(FileExistsError):
        lock_module.create(project, path)


@pytest.mark.parametrize("target", ["protocol/ANALYSIS_PLAN.md",
                                    "data/raw/expansionrx/expansion_data_train.csv",
                                    "src/admetood/model.py"])
def test_any_changed_frozen_input_fails_verification(project, target):
    path = project / "protocol" / "LOCK.json"
    lock_module.create(project, path)
    (project / target).write_text("edited\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since lock"):
        lock_module.verify(project, path)


def test_new_analysis_module_is_allowed_but_can_be_forbidden(project):
    path = project / "protocol" / "LOCK.json"
    lock_module.create(project, path)
    (project / "src" / "admetood" / "later.py").write_text("y = 2\n", encoding="utf-8")
    lock_module.verify(project, path)  # Analysis code is written after the lock.
    with pytest.raises(ValueError, match="New source modules"):
        lock_module.verify(project, path, strict_sources=True)


def test_changed_settings_fail_even_when_files_match(project, monkeypatch):
    path = project / "protocol" / "LOCK.json"
    lock_module.create(project, path)
    monkeypatch.setitem(lock_module.SETTINGS, "equivalence_margin_pp", 10.0)
    with pytest.raises(ValueError, match="settings differ"):
        lock_module.verify(project, path)


def test_missing_frozen_file_is_reported(project):
    (project / "data" / "raw" / "expansionrx" / "expansion_data_test.csv").unlink()
    with pytest.raises(FileNotFoundError, match="expansion_data_test.csv"):
        lock_module.create(project, project / "protocol" / "LOCK.json")


def test_tampered_lock_file_is_rejected(project):
    path = project / "protocol" / "LOCK.json"
    lock_module.create(project, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "something_else"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized lock status"):
        lock_module.verify(project, path)
