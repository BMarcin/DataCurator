"""HTTP + dynamic-batching tests for the MetricX service.

Uses a stub scorer, so no torch / MetricX stack is needed — this exercises the
Starlette surface and the batcher, not the model. Model faithfulness is checked
separately on a GPU box against `metricx/score_metricx.py` (see README).
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from starlette.testclient import TestClient

from server import build_app


def _pair(src, hyp):
    return {"source": src, "hypothesis": hyp}


def test_score_preserves_order_within_request():
    # Stub returns the numeric value of each hypothesis, so we can assert that
    # scores come back aligned to the submitted pairs.
    def stub(pairs):
        return [float(p["hypothesis"]) for p in pairs]

    app = build_app(infer_fn=stub, max_wait_ms=20)
    with TestClient(app) as client:
        r = client.post(
            "/score",
            json={"pairs": [_pair("s", "10"), _pair("s", "20"), _pair("s", "30")]},
        )
        assert r.status_code == 200
        assert r.json()["scores"] == [10.0, 20.0, 30.0]


def test_empty_pairs_short_circuits():
    calls = []

    def stub(pairs):
        calls.append(len(pairs))
        return [0.0] * len(pairs)

    app = build_app(infer_fn=stub)
    with TestClient(app) as client:
        r = client.post("/score", json={"pairs": []})
        assert r.status_code == 200
        assert r.json() == {"scores": []}
    assert calls == []  # inference never invoked for an empty batch


def test_concurrent_requests_coalesce_into_one_batch():
    seen = []

    def stub(pairs):
        seen.append(len(pairs))
        # tiny delay so the wait window definitely captures siblings
        time.sleep(0.01)
        return [float(p["hypothesis"]) for p in pairs]

    # generous window + large max so 16 concurrent singletons merge
    app = build_app(infer_fn=stub, max_batch=64, max_wait_ms=100)
    with TestClient(app) as client:
        def one(i):
            r = client.post("/score", json={"pairs": [_pair("s", str(i))]})
            assert r.status_code == 200
            return r.json()["scores"][0]

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(one, range(16)))

    assert sorted(results) == [float(i) for i in range(16)]  # every request got its own score
    assert max(seen) > 1, f"expected cross-request batching, saw batch sizes {seen}"


def test_inference_error_returns_500():
    def stub(pairs):
        raise RuntimeError("boom")

    app = build_app(infer_fn=stub)
    with TestClient(app) as client:
        r = client.post("/score", json={"pairs": [_pair("s", "h")]})
        assert r.status_code == 500


def test_bad_request_returns_400():
    app = build_app(infer_fn=lambda pairs: [0.0] * len(pairs))
    with TestClient(app) as client:
        # missing "pairs"
        assert client.post("/score", json={"nope": 1}).status_code == 400
        # pair missing hypothesis
        assert client.post(
            "/score", json={"pairs": [{"source": "s"}]}
        ).status_code == 400


def test_health_gates_on_model_load():
    release = threading.Event()

    class StubScorer:
        def score_batch(self, pairs):
            return [1.0] * len(pairs)

    def loader():
        release.wait(5)
        return StubScorer()

    app = build_app(model_loader=loader)
    with TestClient(app) as client:
        # model still loading
        assert client.get("/health").status_code == 503
        assert client.get("/health").json()["model_loaded"] is False
        assert client.post(
            "/score", json={"pairs": [_pair("s", "h")]}
        ).status_code == 503

        release.set()
        for _ in range(100):
            if client.get("/health").status_code == 200:
                break
            time.sleep(0.05)
        assert client.get("/health").json()["model_loaded"] is True
        r = client.post("/score", json={"pairs": [_pair("s", "h")]})
        assert r.status_code == 200
        assert r.json()["scores"] == [1.0]


def test_reference_forwarded_when_accepted():
    seen = []

    def stub(pairs):
        seen.extend(pairs)
        return [0.0] * len(pairs)

    app = build_app(infer_fn=stub, accepts_reference=True)
    with TestClient(app) as client:
        r = client.post(
            "/score",
            json={"pairs": [{"source": "s", "hypothesis": "h", "reference": "r"}]},
        )
        assert r.status_code == 200
    assert seen == [{"source": "s", "hypothesis": "h", "reference": "r"}]


def test_reference_rejected_on_qe_deployment():
    app = build_app(infer_fn=lambda pairs: [0.0] * len(pairs))  # accepts_reference=False
    with TestClient(app) as client:
        r = client.post(
            "/score",
            json={"pairs": [{"source": "s", "hypothesis": "h", "reference": "r"}]},
        )
        assert r.status_code == 400
        # plain QE pairs still work against the same deployment
        assert client.post("/score", json={"pairs": [_pair("s", "h")]}).status_code == 200


def test_non_string_reference_rejected():
    app = build_app(infer_fn=lambda pairs: [0.0] * len(pairs), accepts_reference=True)
    with TestClient(app) as client:
        r = client.post(
            "/score",
            json={"pairs": [{"source": "s", "hypothesis": "h", "reference": 1}]},
        )
        assert r.status_code == 400


def test_health_reports_accepts_reference():
    app = build_app(infer_fn=lambda pairs: [0.0] * len(pairs), accepts_reference=True)
    with TestClient(app) as client:
        assert client.get("/health").json()["accepts_reference"] is True


def test_load_failure_is_reported_and_retried():
    attempts = []

    class StubScorer:
        def score_batch(self, pairs):
            return [1.0] * len(pairs)

    def loader():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("download exploded")
        return StubScorer()

    app = build_app(model_loader=loader, load_retry_s=0.01)
    with TestClient(app) as client:
        # first attempt fails -> still 503, but the error is surfaced
        for _ in range(100):
            if client.get("/health").json()["last_error"]:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("load error never surfaced in /health")
        assert "download exploded" in client.get("/health").json()["last_error"]

        # the retry succeeds and clears the error
        for _ in range(100):
            if client.get("/health").status_code == 200:
                break
            time.sleep(0.02)
        health = client.get("/health").json()
        assert health["model_loaded"] is True
        assert health["last_error"] is None


def test_docs_and_openapi_served():
    app = build_app(infer_fn=lambda pairs: [0.0] * len(pairs))
    with TestClient(app) as client:
        docs = client.get("/docs")
        assert docs.status_code == 200
        assert "swagger-ui" in docs.text
        spec = client.get("/openapi.json").json()
        assert set(spec["paths"]) == {"/score", "/health", "/metrics"}
        pair = spec["components"]["schemas"]["Pair"]
        assert pair["required"] == ["source", "hypothesis"]
        assert "reference" in pair["properties"]


def test_bearer_token_enforced_when_set():
    app = build_app(infer_fn=lambda pairs: [0.0] * len(pairs), token="secret")
    with TestClient(app) as client:
        assert client.post("/score", json={"pairs": []}).status_code == 401
        ok = client.post(
            "/score",
            json={"pairs": []},
            headers={"authorization": "Bearer secret"},
        )
        assert ok.status_code == 200
