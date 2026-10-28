from pathlib import Path


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