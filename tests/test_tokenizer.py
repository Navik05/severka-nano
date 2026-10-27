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