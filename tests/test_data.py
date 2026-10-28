import pytest
import torch

from severka.data import encode_text, load_text, split_text
from severka.tokenizer import CharTokenizer


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

def test_encode_text():
    tok = CharTokenizer(["а", "б", "в"])
    ids = encode_text("абв", tok)
    assert ids.shape == (3,)
    assert ids.dtype == torch.long
    assert ids.tolist() == [0, 1, 2]

def test_encode_text_roundtrip():
    text = "привет, мир!\n"
    tok = CharTokenizer.fit(text)
    assert tok.decode(encode_text(text, tok).tolist()) == text

def test_encode_empty():
    tok = CharTokenizer(["а"])
    ids = encode_text("", tok)
    assert ids.shape == (0,)
    assert ids.dtype == torch.long

def test_encode_unknown_char():
    tok = CharTokenizer(["а", "б"])
    with pytest.raises(KeyError):
        encode_text("в", tok)