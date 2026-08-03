from pathlib import Path


def model_size_mb(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.stat().st_size / (1024 * 1024)
