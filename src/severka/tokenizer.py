class CharTokenizer:
    """Сопоставит каждому символу id"""

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = list(vocab)

    @classmethod
    def fit(cls, text: str) -> CharTokenizer:
        """Строит словарь по тексту"""
        return cls(sorted(set(text)))

    def __len__(self) -> int:
        return len(self.vocab)