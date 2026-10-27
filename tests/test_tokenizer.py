import pytest

from severka.tokenizer import CharTokenizer


def test_fit_is_sorted():
    vocab = CharTokenizer.fit("гвба").vocab
    assert vocab == sorted(vocab)

def test_fit_is_set():
    assert len(CharTokenizer.fit("ааббгг").vocab) == 3

def test_init_copies_list():
    source = ["а", "б"]
    tok = CharTokenizer(source)
    source.append("в")
    assert tok.vocab == ["а", "б"]

def test_encode_return():
    tok = CharTokenizer(["а", "б", "в"])
    assert tok.encode("абв") == [0, 1, 2]
    assert tok.encode("ббв") == [1, 1, 2]

def test_encode_unknown_token():
    tok = CharTokenizer(["а", "б", "в"])
    with pytest.raises(KeyError):
        tok.encode("г")