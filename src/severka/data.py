from pathlib import Path

import torch

from severka.tokenizer import CharTokenizer


def load_text(path: Path) -> str:
    """Читает текст в utf-8"""
    with open(path, encoding="utf-8") as f:
        return f.read()

def split_text(text: str, val_part: float = 0.1) -> tuple[str, str]:
    """Делит текст на train и val"""
    if not 0 < val_part < 1:
        raise ValueError(f"val_part должен быть между 0 и 1, получено {val_part}")

    n = int(len(text) * (1 - val_part))
    return text[:n], text[n:]

def encode_text(text: str, tokenizer: CharTokenizer) -> torch.Tensor:
    """Кодирует текст в одномерный тензор"""
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def get_batch(
        data: torch.Tensor,
        block_size: int,
        batch_size: int,
        generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Возвращает случайные пары (inputs, targets) для обучения"""
    if not data.ndim == 1:
        raise ValueError("data должен быть одномерным тензором")
    if not (block_size > 0 and batch_size > 0):
        raise ValueError("block_size и batch_size должны быть > 0")
    if not len(data) > block_size:
        raise ValueError("длина data должна быть больше block_size")

    high = len(data) - block_size
    starts = torch.randint(high, (batch_size,), generator=generator)
    offsets = torch.arange(block_size)
    idx = starts[:, None] + offsets
    return data[idx], data[idx + 1]