from pathlib import Path


def load_text(path: Path) -> str:
    """Читает текст в utf-8"""
    with open(path, encoding="utf-8") as f:
        return f.read()