"""HTTP + dynamic-batching tests for the LaBSE embedding service.

Uses a stub embedder, so no torch / sentence-transformers stack is needed — this
exercises the Starlette surface and the batcher, not the model. Vector
faithfulness (dim, unit norm, cos≈1 on identical text) is checked separately on
a real service instance (see README).
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from starlette.testclient import TestClient

from server import build_app


def test_embed_preserves_order_within_request():
    # Stub returns [float(text)] so we can assert vectors come back aligned to
    # the submitted texts.
    def stub(texts):
        return [[float(t)] for t in texts]

    app = build_app(infer_fn=stub, dim=1, max_wait_ms=20)
    with TestClient(app) as client:
        r = client.post("/embed", json={"texts": ["10", "20", "30"]})
        assert r.status_code == 200
        assert r.json()["embeddings"] == [[10.0], [20.0], [30.0]]
        assert r.json()["dim"] == 1


def test_empty_texts_short_circuits():
    calls = []

    def stub(texts):
        calls.append(len(texts))
        return [[0.0] for _ in texts]

    app = build_app(infer_fn=stub, dim=1)
    with TestClient(app) as client:
        r = client.post("/embed", json={"texts": []})
        assert r.status_code == 200
        assert r.json() == {"embeddings": [], "dim": 1}
    assert calls == []  # inference never invoked for an empty batch


def test_concurrent_requests_coalesce_into_one_batch():
    seen = []

    def stub(texts):
        seen.append(len(texts))
        time.sleep(0.01)  # tiny delay so the wait window captures siblings
        return [[float(t)] for t in texts]

    app = build_app(infer_fn=stub, dim=1, max_batch=64, max_wait_ms=100)
    with TestClient(app) as client:
        def one(i):
            r = client.post("/embed", json={"texts": [str(i)]})
            assert r.status_code == 200
            return r.json()["embeddings"][0][0]

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(one, range(16)))

    assert sorted(results) == [float(i) for i in range(16)]
    assert max(seen) > 1, f"expected cross-request batching, saw batch sizes {seen}"


def test_inference_error_returns_500():
    def stub(texts):
        raise RuntimeError("boom")

    app = build_app(infer_fn=stub, dim=1)
    with TestClient(app) as client:
        r = client.post("/embed", json={"texts": ["hi"]})
        assert r.status_code == 500


def test_bad_request_returns_400():
    app = build_app(infer_fn=lambda texts: [[0.0] for _ in texts], dim=1)
    with TestClient(app) as client:
        # missing "texts"
        assert client.post("/embed", json={"nope": 1}).status_code == 400
        # non-string item
        assert client.post("/embed", json={"texts": ["ok", 5]}).status_code == 400


def test_health_gates_on_model_load():
    release = threading.Event()

    class StubEmbedder:
        dim = 3

        def embed_batch(self, texts):
            return [[1.0, 2.0, 3.0] for _ in texts]

    def loader():
        release.wait(5)
        return StubEmbedder()

    app = build_app(model_loader=loader)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/health").json()["model_loaded"] is False
        assert client.post("/embed", json={"texts": ["h"]}).status_code == 503

        release.set()
        for _ in range(100):
            if client.get("/health").status_code == 200:
                break
            time.sleep(0.05)
        health = client.get("/health").json()
        assert health["model_loaded"] is True
        assert health["dim"] == 3
        r = client.post("/embed", json={"texts": ["h"]})
        assert r.status_code == 200
        assert r.json() == {"embeddings": [[1.0, 2.0, 3.0]], "dim": 3}


def test_health_reports_dim():
    app = build_app(infer_fn=lambda texts: [[0.0] for _ in texts], dim=768)
    with TestClient(app) as client:
        assert client.get("/health").json()["dim"] == 768


def test_load_failure_is_reported_and_retried():
    attempts = []

    class StubEmbedder:
        dim = 1

        def embed_batch(self, texts):
            return [[1.0] for _ in texts]

    def loader():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("download exploded")
        return StubEmbedder()

    app = build_app(model_loader=loader, load_retry_s=0.01)
    with TestClient(app) as client:
        for _ in range(100):
            if client.get("/health").json()["last_error"]:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("load error never surfaced in /health")
        assert "download exploded" in client.get("/health").json()["last_error"]

        for _ in range(100):
            if client.get("/health").status_code == 200:
                break
            time.sleep(0.02)
        health = client.get("/health").json()
        assert health["model_loaded"] is True
        assert health["last_error"] is None


def test_docs_and_openapi_served():
    app = build_app(infer_fn=lambda texts: [[0.0] for _ in texts], dim=1)
    with TestClient(app) as client:
        docs = client.get("/docs")
        assert docs.status_code == 200
        assert "swagger-ui" in docs.text
        spec = client.get("/openapi.json").json()
        assert set(spec["paths"]) == {"/embed", "/health", "/metrics"}
        assert spec["paths"]["/embed"]["post"]["requestBody"]["required"] is True


def test_bearer_token_enforced_when_set():
    app = build_app(infer_fn=lambda texts: [[0.0] for _ in texts], dim=1, token="secret")
    with TestClient(app) as client:
        assert client.post("/embed", json={"texts": []}).status_code == 401
        ok = client.post(
            "/embed",
            json={"texts": []},
            headers={"authorization": "Bearer secret"},
        )
        assert ok.status_code == 200
