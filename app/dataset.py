from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import SETTINGS
from .text_cleaning import clean_text


@dataclass(frozen=True)
class Document:
    doc_id: str
    label: str
    cleaned_text: str
    raw_preview: str


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="latin-1", errors="ignore")


def load_documents(data_root: Path | None = None) -> list[Document]:
    root = data_root or SETTINGS.data_root
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    documents: list[Document] = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for file_path in sorted(label_dir.iterdir()):
            if not file_path.is_file():
                continue
            raw = _safe_read_text(file_path)
            cleaned = clean_text(raw)
            if len(cleaned.split()) < SETTINGS.min_clean_tokens:
                continue
            documents.append(
                Document(
                    doc_id=f"{label}/{file_path.name}",
                    label=label,
                    cleaned_text=cleaned,
                    raw_preview=raw[:300].replace("\n", " ").strip(),
                )
            )
    return documents


def document_texts(documents: Iterable[Document]) -> list[str]:
    return [doc.cleaned_text for doc in documents]

