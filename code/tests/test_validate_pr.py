"""Focused tests for the repository PR validator helpers."""

from __future__ import annotations

import zipfile

import pytest

from scripts.validate_pr import (
    ValidationError,
    assert_wheel_members,
    find_wheel,
    parse_passed_test_count,
)


def test_parse_passed_test_count_reads_pytest_summary() -> None:
    assert parse_passed_test_count("...\n162 passed in 7.62s\n") == 162


def test_parse_passed_test_count_rejects_malformed_summary() -> None:
    with pytest.raises(ValidationError, match="parseable"):
        parse_passed_test_count("162 tests passed in 7.62s\n")


def test_parse_passed_test_count_rejects_missing_summary() -> None:
    with pytest.raises(ValidationError, match="parseable"):
        parse_passed_test_count("no tests ran\n")


def test_find_wheel_rejects_empty_directory(tmp_path) -> None:
    with pytest.raises(ValidationError, match="exactly one wheel"):
        find_wheel(tmp_path)


def test_find_wheel_rejects_multiple_wheels(tmp_path) -> None:
    (tmp_path / "first.whl").touch()
    (tmp_path / "second.whl").touch()

    with pytest.raises(ValidationError, match="exactly one wheel"):
        find_wheel(tmp_path)


def write_wheel(path, members: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, b"content")


def test_assert_wheel_members_rejects_missing_member(tmp_path) -> None:
    wheel = tmp_path / "missing.whl"
    write_wheel(wheel, ["package/__init__.py"])

    with pytest.raises(ValidationError, match="missing wheel members"):
        assert_wheel_members(wheel, frozenset({"package/__init__.py", "package/core.py"}))


def test_assert_wheel_members_rejects_forbidden_bytecode(tmp_path) -> None:
    wheel = tmp_path / "bytecode.whl"
    write_wheel(wheel, ["package/__init__.py", "package/__pycache__/core.cpython-313.pyc"])

    with pytest.raises(ValidationError, match="forbidden wheel members"):
        assert_wheel_members(wheel, frozenset({"package/__init__.py"}))
