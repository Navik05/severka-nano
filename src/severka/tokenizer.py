from collections.abc import Iterable


class CharTokenizer:
    """Сопоставит каждому символу id"""

    def __init__(self, vocab: list[str]) -> None:
        self.vocab: list[str] = list(vocab)
        self.stoi: dict[str, int] = {token: i for i, token in enumerate(self.vocab)}
        self.itos: dict[int, str] = {id: token for id, token in enumerate(self.vocab)}

    @classmethod
    def fit(cls, text: str) -> CharTokenizer:
        """Строит словарь по тексту"""
        return cls(sorted(set(text)))

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Переводит текст в id посимвольно"""
        return [self.stoi[token] for token in text]

    def decode(self, ids: Iterable[int]) -> str:
        """Переводит последовательность id в строку"""
        return "".join(self.itos[id] for id in ids)