import pytest

from severka.data import load_text


def test_load_text(tmp_path):
    path = tmp_path / "raw.txt"
    path.write_text("Первое\nслово", encoding="utf-8")
    assert load_text(path) == "Первое\nслово"

def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text(tmp_path / "нетфайла.txt")