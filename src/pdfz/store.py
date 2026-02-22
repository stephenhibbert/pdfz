from __future__ import annotations

import json
import os
from pathlib import Path

from pdfz.models import PDFDocument

_DATA_DIR = Path(os.environ.get("PDFZ_DATA_DIR", Path(__file__).parent.parent.parent / "data"))
DEFAULT_DB_PATH = _DATA_DIR / "documents.json"


class DocumentStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self.db_path.write_text("[]")

    def _load(self) -> list[PDFDocument]:
        raw = json.loads(self.db_path.read_text())
        return [PDFDocument.model_validate(doc) for doc in raw]

    def _save(self, docs: list[PDFDocument]) -> None:
        self.db_path.write_text(
            json.dumps([doc.model_dump(mode="json") for doc in docs], indent=2)
        )

    def add(self, doc: PDFDocument) -> PDFDocument:
        docs = self._load()
        docs.append(doc)
        self._save(docs)
        return doc

    def list_all(self) -> list[PDFDocument]:
        return self._load()

    def get(self, document_id: str) -> PDFDocument | None:
        for doc in self._load():
            if doc.id == document_id:
                return doc
        return None

    def find_by_hash(self, content_hash: str) -> PDFDocument | None:
        for doc in self._load():
            if doc.content_hash == content_hash:
                return doc
        return None
