"""Embeddings and the vector index (Part B §14, §43, §58).

Part B §14 forbids relying on embeddings alone, and this module is the semantic *component* of a
hybrid ranker, not the ranker.

Two embedders, for two jobs:

* `HashingEmbedder` — a deterministic character-n-gram hashing embedder. Not a stand-in for a
  learned model: it is a real, if blunt, semantic index that captures morphological and lexical
  overlap, needs no network, and is exactly reproducible. That last property is what makes it the
  right default for a benchmark whose entire claim rests on the individual agent being frozen
  (§20) — a hosted embedding model can be revised underneath a campaign, and the campaign would
  have no way to notice.
* `ProviderEmbedder` — a complete adapter for a hosted embedding API, mocked in tests
  (Part A §A1.8).

The index itself is `numpy` brute force on SQLite and pgvector where installed. Brute force over a
few hundred thousand vectors is milliseconds; §59's millions need pgvector, and the abstraction is
here so that is a deployment choice rather than a rewrite.
"""

from __future__ import annotations

import abc
import hashlib
import re
import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.persistence.models import Artifact, ArtifactEmbedding

DEFAULT_DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(abc.ABC):
    """Text to vector. Named and versioned, because a re-embedding under a different model must
    not silently replace vectors a previous experiment's retrieval was computed against."""

    model: str = "base"
    dim: int = DEFAULT_DIM

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalised."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Deterministic hashed character-n-grams plus whole words.

    Character n-grams matter for this corpus: agent-written artifacts share stems and identifiers
    more than they share vocabulary, and a word-only representation misses `retry`/`retries` and
    every hyphenated identifier. Whole words are added at a higher weight so exact term matches
    still dominate.

    Uses `hashlib`, not `hash()`: Python randomises string hashing per process, which would make
    two runs of the same campaign produce different vectors (§46).
    """

    def __init__(self, dim: int = DEFAULT_DIM, *, ngram: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.model = f"hashing-{dim}-{ngram[0]}{ngram[1]}"
        self._lo, self._hi = ngram

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = _TOKEN.findall((text or "").lower())
            for word in words:
                out[row, self._bucket(f"w:{word}")] += 2.0
                padded = f" {word} "
                for n in range(self._lo, self._hi + 1):
                    for i in range(max(0, len(padded) - n + 1)):
                        out[row, self._bucket(padded[i : i + n])] += 1.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class ProviderEmbedder(Embedder):
    """A hosted embedding API (Part B §19, §43). Complete adapter, mocked in tests."""

    def __init__(self, *, model: str = "text-embedding-3-small", dim: int = 1536,
                 base_url: str = "https://api.openai.com", api_key: str | None = None,
                 client=None):
        self.model = model
        self.dim = dim
        self._base_url = base_url
        self._api_key = api_key
        self._client = client

    def embed(self, texts: list[str]) -> np.ndarray:
        import httpx

        client = self._client or httpx.Client(timeout=60.0)
        response = client.post(
            f"{self._base_url.rstrip('/')}/v1/embeddings",
            json={"model": self.model, "input": texts},
            headers={"authorization": f"Bearer {self._api_key}"} if self._api_key else {},
        )
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda d: d.get("index", 0))
        out = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


_DEFAULT: Embedder | None = None


def default_embedder() -> Embedder:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = HashingEmbedder()
    return _DEFAULT


def set_default_embedder(embedder: Embedder) -> None:
    global _DEFAULT
    _DEFAULT = embedder


def artifact_text(artifact: Artifact) -> str:
    """What gets embedded.

    The title twice: it is the most information-dense field an agent writes, and repeating it is a
    cheap, honest weighting rather than a learned one.
    """
    return f"{artifact.title}\n{artifact.title}\n{artifact.body}"


def index_artifact(
    session: Session, artifact: Artifact, embedder: Embedder | None = None
) -> ArtifactEmbedding:
    """Embed one artifact, replacing any vector for the same model.

    Keyed by (artifact, model): a new model adds a row rather than overwriting, so retrieval
    computed under an older model stays reproducible (§46).
    """
    embedder = embedder or default_embedder()
    vector = embedder.embed_one(artifact_text(artifact))

    existing = session.execute(
        select(ArtifactEmbedding).where(
            ArtifactEmbedding.artifact_id == artifact.id,
            ArtifactEmbedding.model == embedder.model,
        )
    ).scalar_one_or_none()
    if existing is not None:
        existing.vector = vector
        existing.dim = embedder.dim
        existing.norm = float(np.linalg.norm(vector))
        session.flush()
        return existing

    row = ArtifactEmbedding(
        artifact_id=artifact.id, model=embedder.model, dim=embedder.dim,
        vector=vector, norm=float(np.linalg.norm(vector)),
    )
    session.add(row)
    session.flush()
    return row


def index_missing(session: Session, workspace_id: uuid.UUID, *, embedder: Embedder | None = None,
                  batch: int = 256) -> int:
    """Embed everything in a workspace that has no vector under this model.

    Batched because §58 requires batch embeddings: one call per artifact is the difference between
    a workspace that can be indexed and one that cannot.
    """
    embedder = embedder or default_embedder()
    indexed = set(
        session.execute(
            select(ArtifactEmbedding.artifact_id)
            .join(Artifact, Artifact.id == ArtifactEmbedding.artifact_id)
            .where(Artifact.workspace_id == workspace_id,
                   ArtifactEmbedding.model == embedder.model)
        ).scalars()
    )
    pending = [
        a for a in session.execute(
            select(Artifact).where(Artifact.workspace_id == workspace_id)
        ).scalars()
        if a.id not in indexed
    ]
    count = 0
    for start in range(0, len(pending), batch):
        chunk = pending[start : start + batch]
        vectors = embedder.embed([artifact_text(a) for a in chunk])
        for artifact, vector in zip(chunk, vectors, strict=True):
            session.add(ArtifactEmbedding(
                artifact_id=artifact.id, model=embedder.model, dim=embedder.dim,
                vector=vector, norm=float(np.linalg.norm(vector)),
            ))
            count += 1
        session.flush()
    return count


@dataclass
class VectorHit:
    artifact_id: uuid.UUID
    score: float


def search(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    query: str,
    embedder: Embedder | None = None,
    limit: int = 50,
    candidate_ids: set[uuid.UUID] | None = None,
) -> list[VectorHit]:
    """Cosine similarity over the workspace's vectors.

    `candidate_ids` restricts the search to a set the caller has already filtered — which is how
    the arm policies stay enforced: the vector index must never be a way around an ablation
    (§21).
    """
    embedder = embedder or default_embedder()
    stmt = (
        select(ArtifactEmbedding.artifact_id, ArtifactEmbedding.vector)
        .join(Artifact, Artifact.id == ArtifactEmbedding.artifact_id)
        .where(Artifact.workspace_id == workspace_id,
               ArtifactEmbedding.model == embedder.model)
    )
    if candidate_ids is not None:
        if not candidate_ids:
            return []
        stmt = stmt.where(ArtifactEmbedding.artifact_id.in_(candidate_ids))

    rows = session.execute(stmt).all()
    if not rows:
        return []

    ids = [r[0] for r in rows]
    matrix = np.vstack([np.asarray(r[1], dtype=np.float32) for r in rows])
    q = embedder.embed_one(query)
    scores = matrix @ q

    order = np.argsort(-scores)[:limit]
    return [VectorHit(ids[i], float(scores[i])) for i in order if scores[i] > 0]
