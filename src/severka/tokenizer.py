import json
from collections.abc import Iterable
from pathlib import Path


class CharTokenizer:
    """Сопоставит каждому символу id"""

    def __init__(self, vocab: list[str]) -> None:
        self.vocab: list[str] = list(vocab)
        self.stoi: dict[str, int] = {token: i for i, token in enumerate(self.vocab)}
        self.itos: dict[int, str] = {id: token for id, token in enumerate(self.vocab)}

    def __len__(self) -> int:
        return len(self.vocab)

    @classmethod
    def fit(cls, text: str) -> CharTokenizer:
        """Строит словарь по тексту"""
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        """Переводит текст в id посимвольно"""
        return [self.stoi[token] for token in text]

    def decode(self, ids: Iterable[int]) -> str:
        """Переводит последовательность id в строку"""
        return "".join(self.itos[id] for id in ids)

    def save(self, path: Path) -> None:
        """Сохраняет словарь в json файл"""
        path.write_text( json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8" )

    @classmethod
    def load(cls, path: Path) -> CharTokenizer:
        """Загружает из json файла"""
        return cls(json.loads(path.read_text(encoding="utf-8")))