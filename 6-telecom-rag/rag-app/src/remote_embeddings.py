"""
Embeddings that live in ANOTHER container.

Instead of loading the model inside this app (HuggingFaceEmbeddings), we call the
embedding-service over HTTP. Same model, same vectors, different place.
"""
import json
import os
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import httpx  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback for minimal envs
    httpx = None

try:
    from langchain_core.embeddings import Embeddings  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback for minimal envs
    class Embeddings:  # type: ignore[no-redef]
        """Fallback base class when langchain-core is unavailable."""

        pass

EMBEDDING_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001")

class RemoteEmbeddings(Embeddings):
    def __init__(self, url: str = EMBEDDING_URL, batch_size: int = 64):
        self.url = url.rstrip("/")
        self.batch_size = batch_size

    def _post_json(self, url: str, payload: dict) -> dict:
        if httpx is not None:
            resp = httpx.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()

        data = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Embedding service request failed ({exc.code}): {body or exc.reason}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Embedding service unavailable: {exc.reason}") from exc

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            payload = self._post_json(f"{self.url}/embed", {"texts": batch})
            vectors.extend(payload["embeddings"])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]