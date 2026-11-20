"""A tiny key-value store in one JSON file (no server, easy to read). Values can be anything JSON can hold."""
import json
import os
from pathlib import Path


DEFAULT_CORPUS_PATH = Path(__file__).with_name("corpus.txt")


def load_corpus(path: str | Path | None = None) -> str:
    """Load a UTF-8 corpus file, defaulting to the bundled short-story corpus."""
    corpus_path = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    return corpus_path.read_text(encoding="utf-8")


class Store:
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.environ.get("DATA_PATH", "data/app.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}")

    def _read(self) -> dict:
        return json.loads(self.path.read_text() or "{}")

    def put(self, key: str, value) -> None:
        data = self._read()
        data[key] = value
        self.path.write_text(json.dumps(data, indent=2))

    def get(self, key: str, default=None):
        return self._read().get(key, default)

    def all(self) -> dict:
        return dict(sorted(self._read().items()))
