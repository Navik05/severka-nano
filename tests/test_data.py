import pytest

from severka.data import load_text, split_text


def test_load_text(tmp_path):
    path = tmp_path / "raw.txt"
    path.write_text("Первое\nслово", encoding="utf-8")
    assert load_text(path) == "Первое\nслово"

def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text(tmp_path / "нетфайла.txt")

def test_split_default():
    train, val = split_text("абвгдеёжзи")
    assert len(train) == 9
    assert len(val) == 1

def test_split_text():
    text = "абвгдеёжзи"
    train, val = split_text(text,0.4)
    assert train == "абвгде"
    assert val == "ёжзи"
    assert train + val == text

@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.1])
def test_split_invalid(bad):
    with pytest.raises(ValueError):
        split_text("абвгдеёжзи", bad)