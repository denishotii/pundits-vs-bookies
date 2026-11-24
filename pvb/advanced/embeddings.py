"""Frozen, local article embeddings and their private per-article cache."""

from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import pandas as pd

from ..config import ROOT
from .preprocessing import PRIMARY_TEXT_COLUMNS, TEXT_REPRESENTATIONS, UNICODE_FORM, prepare_documents

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
MODEL_RELEASE = "2021-08-30"
MODEL_RELEASE_SOURCE = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/commit/acbb28c8aa70f5503c85d6b90e8cd65606993a20"
CACHE_ROOT = ROOT / "data" / "cache" / "contextual"
INFERENCE_BATCH_SIZE = 64
DOCUMENT_BATCH_SIZE = 64
REPRESENTATION = {
    "schema": 1,
    "sections": PRIMARY_TEXT_COLUMNS,
    "normalization": f"{UNICODE_FORM}, collapse whitespace per section",
    "chunking": "Nonoverlapping consecutive token IDs within each section, reserving special-token slots",
    "pooling": "L2-normalized chunks, weighted by content token count, then L2-normalize the article",
    "empty_document": "All-zero article vector",
    "dtype": "float32",
}


class EmbeddingUnavailable(RuntimeError):
    """The optional local inference stack or pinned weights could not load."""


class MiniLMEncoder:
    """Run the pinned Sentence Transformer directly on prepared token IDs."""

    def __init__(self, model_cache: Path, *, local_files_only: bool = False):
        start = perf_counter()
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except (ImportError, OSError, RuntimeError) as error:
            raise EmbeddingUnavailable(
                f"{error}. Install the optional stack with "
                r".\.venv\Scripts\python.exe -m pip install -r requirements-contextual.txt"
            ) from error

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = SentenceTransformer(
                MODEL_NAME, revision=MODEL_REVISION, device=self.device, cache_folder=str(model_cache),
                local_files_only=local_files_only, trust_remote_code=False, token=False,
                model_kwargs={"use_safetensors": True},
            ).eval().requires_grad_(False)
        except (OSError, RuntimeError, ValueError) as error:
            raise EmbeddingUnavailable(f"Could not load {MODEL_NAME}@{MODEL_REVISION}: {error}") from error
        self.tokenizer = self.model.tokenizer
        self.max_input_tokens = min(self.model.max_seq_length, self.tokenizer.model_max_length)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.config = {
            "model": MODEL_NAME, "revision": MODEL_REVISION, "dimensions": self.dimension,
            "tokenizer": {
                "class": type(self.tokenizer).__name__, "is_fast": self.tokenizer.is_fast,
                "model_max_length": self.tokenizer.model_max_length, "max_input_tokens": self.max_input_tokens,
                "special_tokens_per_chunk": self.tokenizer.num_special_tokens_to_add(pair=False),
                "special_token_ids": self.tokenizer.all_special_ids,
                "padding_side": self.tokenizer.padding_side, "vocab_size": self.tokenizer.vocab_size,
                "do_lower_case": self.tokenizer.do_lower_case,
            },
            "versions": {package: version(package) for package in (
                "torch", "sentence-transformers", "transformers", "tokenizers", "huggingface-hub", "safetensors",
            )},
        }
        self.load_seconds = perf_counter() - start

    @property
    def runtime(self) -> dict:
        return {
            "device": self.device,
            "cuda_version": self._torch.version.cuda if self.device == "cuda" else None,
            "cuda_device": self._torch.cuda.get_device_name() if self.device == "cuda" else None,
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "encoder_load_seconds": self.load_seconds,
        }

    def encode_chunks(self, input_ids: list[list[int]]) -> np.ndarray:
        padded = self.tokenizer.pad([{"input_ids": ids} for ids in input_ids], return_tensors="pt")
        features = {name: tensor.to(self.device) for name, tensor in padded.items()}
        with self._torch.inference_mode():
            return self.model(features)["sentence_embedding"].float().cpu().numpy()


def tokenize_chunks(sections, tokenizer, max_input_tokens: int) -> list[tuple[list[int], int]]:
    """Return model inputs and content lengths without decoding or truncating IDs."""
    width = max_input_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if width < 1:
        raise ValueError("The model input length must allow at least one content token")
    chunks = []
    for section in sections:
        if not section:
            continue
        ids = tokenizer.encode(section, add_special_tokens=False, truncation=False, verbose=False)
        for start in range(0, len(ids), width):
            content = ids[start:start + width]
            model_input = tokenizer.build_inputs_with_special_tokens(content)
            if len(model_input) > max_input_tokens:
                raise ValueError("Tokenizer added more special tokens than expected")
            chunks.append((model_input, len(content)))
    return chunks


def pool_chunks(vectors: np.ndarray, lengths: list[int]) -> np.ndarray:
    """Normalize the chunks, weight by content length and unit-normalize the result."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) != len(lengths) or not np.isfinite(vectors).all():
        raise ValueError("Chunk embeddings must be finite and align with their token lengths")
    if len(vectors) == 0:
        return np.zeros(vectors.shape[1], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0) or np.any(np.asarray(lengths) <= 0):
        raise ValueError("Nonempty chunks require nonzero embeddings and positive token lengths")
    article = np.average(vectors / norms, axis=0, weights=lengths)
    norm = np.linalg.norm(article)
    return (article / norm if norm else article).astype(np.float32)


def _fingerprint(value) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _encode_documents(documents, encoder) -> tuple[np.ndarray, np.ndarray]:
    chunks, lengths, counts = [], [], []
    for sections in documents:
        items = tokenize_chunks(sections, encoder.tokenizer, encoder.max_input_tokens)
        counts.append(len(items))
        for tokens, length in items:
            chunks.append(tokens)
            lengths.append(length)
    vectors = np.empty((len(chunks), encoder.dimension), dtype=np.float32)
    # Batch similar lengths together and restore document/chunk order before pooling.
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]))
    for start in range(0, len(order), INFERENCE_BATCH_SIZE):
        batch = order[start:start + INFERENCE_BATCH_SIZE]
        encoded = encoder.encode_chunks([chunks[i] for i in batch])
        if encoded.shape != (len(batch), encoder.dimension):
            raise ValueError("Encoder output shape does not match chunk count and embedding dimensions")
        vectors[batch] = encoded
    articles = np.empty((len(counts), encoder.dimension), dtype=np.float32)
    start = 0
    for row, count in enumerate(counts):
        end = start + count
        articles[row] = pool_chunks(vectors[start:end], lengths[start:end])
        start = end
    return articles, np.asarray(counts, dtype=int)


@dataclass
class EmbeddingResult:
    vectors: pd.DataFrame
    statistics: dict


def embed_articles(
    sections: pd.DataFrame, encoder, cache_path: Path, *, representation: str = "primary",
) -> EmbeddingResult:
    """Embed only the supplied article IDs and cache vectors by content and encoder."""
    if sections.empty or not sections.index.is_unique or sections.index.hasnans:
        raise ValueError("Embedding inputs require nonempty, unique, nonmissing article IDs")
    start = perf_counter()
    columns = TEXT_REPRESENTATIONS[representation]
    normalized = pd.DataFrame({
        name: prepare_documents(sections.reindex(columns=[name]), representation) for name in columns
    })
    contents = list(normalized.itertuples(index=False, name=None))
    signature = _fingerprint((encoder.config, {**REPRESENTATION, "sections": columns}))
    keys = [_fingerprint((str(article_id), _fingerprint(content), signature))
            for article_id, content in zip(sections.index, contents, strict=True)]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.empty((len(keys), encoder.dimension), dtype=np.float32)
    chunk_counts = np.empty(len(keys), dtype=int)
    generation_devices = Counter()
    missing = []
    inference_seconds = 0.0
    with closing(sqlite3.connect(cache_path, timeout=30)) as database:
        with database:
            database.execute("""
                CREATE TABLE IF NOT EXISTS article_embeddings (
                    cache_key TEXT PRIMARY KEY, vector BLOB NOT NULL,
                    chunks INTEGER NOT NULL, device TEXT NOT NULL
                )
            """)
        for i, key in enumerate(keys):
            row = database.execute(
                "SELECT vector, chunks, device FROM article_embeddings WHERE cache_key = ?", (key,),
            ).fetchone()
            if row is not None and len(row[0]) == encoder.dimension * 4 and row[1] >= 0:
                vector = np.frombuffer(row[0], dtype="<f4")
                if np.isfinite(vector).all():
                    vectors[i], chunk_counts[i] = vector, row[1]
                    generation_devices[row[2]] += 1
                    continue
            missing.append(i)
        for offset in range(0, len(missing), DOCUMENT_BATCH_SIZE):
            batch = missing[offset:offset + DOCUMENT_BATCH_SIZE]
            begin = perf_counter()
            articles, counts = _encode_documents([contents[i] for i in batch], encoder)
            inference_seconds += perf_counter() - begin
            vectors[batch], chunk_counts[batch] = articles, counts
            generation_devices[encoder.device] += len(batch)
            with database:
                database.executemany(
                    "INSERT OR REPLACE INTO article_embeddings VALUES (?, ?, ?, ?)",
                    [(keys[i], vector.astype("<f4", copy=False).tobytes(), int(count), encoder.device)
                     for i, vector, count in zip(batch, articles, counts, strict=True)],
                )

    return EmbeddingResult(pd.DataFrame(vectors, index=sections.index), {
        "articles": len(sections),
        "coverage": float((chunk_counts > 0).mean()),
        "empty_articles": int((chunk_counts == 0).sum()),
        "empty_sections": normalized.eq("").sum().to_dict(),
        "dimensions": encoder.dimension,
        "total_chunks": int(chunk_counts.sum()),
        "new_chunks": int(chunk_counts[missing].sum()),
        "mean_chunks": float(chunk_counts.mean()),
        "p95_chunks": float(np.quantile(chunk_counts, 0.95)),
        "max_chunks": int(chunk_counts.max()),
        "cache_hits": len(keys) - len(missing),
        "cache_misses": len(missing),
        "article_cache_key_fingerprint": _fingerprint(keys),
        "generation_devices": dict(generation_devices),
        "inference_and_pooling_seconds": inference_seconds,
        "total_seconds": perf_counter() - start,
    })
