import pytest
import torch

from severka.data import encode_text, get_batch, load_text, split_text
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

def make_generator(seed: int = 0) -> torch.Generator:
    """ отдельный генератор, чтобы тесты не зависели друг от друга """
    return torch.Generator().manual_seed(seed)

def test_get_batch_shapes_and_dtype():
    data = torch.arange(20, dtype=torch.long)
    x, y = get_batch(data, block_size=4, batch_size=3, generator=make_generator())
    assert x.shape == (3, 4)
    assert y.shape == (3, 4)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

def test_get_batch_is_shifted_by_one():
    """ y = х со сдвигом на один """
    data = torch.arange(10, dtype=torch.long)
    x, y = get_batch(data, 1, 4, generator=make_generator())
    assert x.shape == (4, 1)
    assert torch.equal(y, x + 1)


def test_get_batch_rows_continuous_data():
    """ х и y - непрерывный кусок data(только y со сдвигом) """
    data = torch.arange(20, dtype=torch.long)
    block_size, batch_size = 4, 3
    x, y = get_batch(data, block_size, batch_size, generator=make_generator())
    for row in range(batch_size):
        start = int(x[row, 0])
        assert torch.equal(x[row], torch.arange(start, start + block_size))
        assert torch.equal(y[row], torch.arange(start + 1, start + block_size + 1))

def test_get_batch_stays_in_bounds():
    """ выход за границы тензора """
    data = torch.arange(10, dtype=torch.long)
    x, y = get_batch(data, block_size=4, batch_size=1, generator=make_generator())
    assert x.min().item() >= 0
    assert x.max().item() < len(data)
    assert y.min().item() >= 0
    assert y.max().item() < len(data)

def test_get_batch_minimal_length():
    """ допустим ровно один старт """
    data = torch.arange(5, dtype=torch.long)
    x, y = get_batch(data, 4, 2, generator=make_generator())
    expected_x = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    expected_y = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    assert torch.equal(x, expected_x)
    assert torch.equal(y, expected_y)

def test_get_batch_reproducible():
    """ одинаковый seed даёт одинаковые батчи """
    data = torch.arange(20, dtype=torch.long)
    x1, y1 = get_batch(data, 4, 3, generator=make_generator(42))
    x2, y2 = get_batch(data, 4, 3, generator=make_generator(42))
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


@pytest.mark.parametrize("data_len", [0, 4])
def test_get_batch_too_short_data(data_len):
    """ len(data) <= block_size """
    data = torch.arange(data_len, dtype=torch.long)
    with pytest.raises(ValueError):
        get_batch(data, 4, 1, generator=make_generator())


@pytest.mark.parametrize(
    ("block_size", "batch_size"),
    [(0, 1), (-1, 1), (4, 0), (4, -1)]
)
def test_get_batch_invalid_sizes(block_size, batch_size):
    """ нулевые и отрицательные размеры для block_size и batch_size """
    data = torch.arange(20, dtype=torch.long)
    with pytest.raises(ValueError):
        get_batch(data, block_size, batch_size, generator=make_generator())