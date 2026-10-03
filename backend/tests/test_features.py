import os
import sys
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["RATE_LIMIT_PER_MIN"] = "1000"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import features  # noqa: E402

app = FastAPI()
features.install(app)
client = TestClient(app)


def test_speech_analytics_counts_fillers_and_pace():
    r = client.post("/api/speech/analyze", json={"text": "Um, I basically built an API. You know, it was fast.", "seconds": 6})
    j = r.json()
    assert j["filler_total"] == 3
    assert j["words"] == 11
    assert j["wpm"] == 110 and j["pace"] == "good pace"


def test_speech_without_duration_has_no_wpm():
    j = client.post("/api/speech/analyze", json={"text": "Hello there"}).json()
    assert j["wpm"] is None


def test_register_login_and_history_flow():
    r = client.post("/api/auth/register", json={"email": "a@b.com", "password": "secret1"})
    assert r.status_code == 200
    token = r.json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/auth/register", json={"email": "a@b.com", "password": "secret1"}).status_code == 409
    assert client.post("/api/auth/login", json={"email": "a@b.com", "password": "wrong"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "a@b.com", "password": "secret1"}).status_code == 200
    save = client.post("/api/history/save", headers=h, json={"role": "Dev", "score": 82, "recommendation": "Hire", "wpm": 120, "fillers": 2})
    assert save.json() == {"saved": True}
    items = client.get("/api/history", headers=h).json()["items"]
    assert len(items) == 1 and items[0]["score"] == 82


def test_history_requires_login():
    assert client.get("/api/history").status_code == 401


def test_weak_password_and_bad_email_rejected():
    assert client.post("/api/auth/register", json={"email": "x@y.com", "password": "123"}).status_code == 400
    assert client.post("/api/auth/register", json={"email": "not-an-email", "password": "secret1"}).status_code == 400


def test_bulk_rank_puts_relevant_resume_first():
    jd = "Python backend developer with FastAPI, Docker, SQL and REST API experience"
    files = [
        ("files", ("chef.txt", b"Experienced chef. Cooking, menu planning, kitchen management, baking.", "text/plain")),
        ("files", ("dev.txt", b"Python developer. Built FastAPI REST API services, Docker containers, SQL databases.", "text/plain")),
    ]
    res = client.post("/api/bulk-rank", data={"job_description": jd}, files=files).json()["results"]
    assert res[0]["name"] == "dev.txt" and res[0]["rank"] == 1
    assert res[0]["score"] > res[1]["score"]


def test_pdf_report_is_valid_pdf():
    r = client.post("/api/report/pdf", json={"role": "Dev", "overall": 80, "recommendation": "Hire",
                                             "strengths": ["Clear <b>communication</b>"], "transcript": [{"q": "Q?", "a": "A.", "feedback": "ok"}]})
    assert r.status_code == 200 and r.content.startswith(b"%PDF")


# ───────────── owner email alerts ─────────────
def test_login_and_register_trigger_alert_without_leaking_password(monkeypatch):
    sent = []
    monkeypatch.setattr(features, "notify_owner", lambda subject, body: sent.append((subject, body)))
    client.post("/api/auth/register", json={"email": "alert@x.com", "password": "TopSecret9"})
    client.post("/api/auth/login", json={"email": "alert@x.com", "password": "TopSecret9"})
    client.post("/api/auth/login", json={"email": "alert@x.com", "password": "wrong-pass"})  # failed: no alert
    assert [s for s, _ in sent] == ["[AI Recruitment] New registration: alert@x.com",
                                     "[AI Recruitment] User login: alert@x.com"]
    assert all("TopSecret9" not in b and "wrong-pass" not in b for _, b in sent)


def test_notify_owner_uses_resend_api(monkeypatch):
    calls = []

    class R:
        ok = True

    monkeypatch.setenv("NOTIFY_TO", "me@example.com")
    monkeypatch.setenv("RESEND_API_KEY", "re_test")
    monkeypatch.setattr(features.requests, "post", lambda url, **kw: calls.append((url, kw)) or R())
    features.notify_owner("subj", "body")
    url, kw = calls[0]
    assert url == "https://api.resend.com/emails"
    assert kw["json"]["to"] == ["me@example.com"] and kw["headers"]["Authorization"] == "Bearer re_test"


def test_notify_owner_never_raises_and_is_silent_when_unconfigured(monkeypatch):
    monkeypatch.delenv("NOTIFY_TO", raising=False)
    features.notify_owner("s", "b")  # no config: no-op
    monkeypatch.setenv("NOTIFY_TO", "me@example.com")
    monkeypatch.setenv("RESEND_API_KEY", "re_test")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(features.requests, "post", boom)
    features.notify_owner("s", "b")  # must swallow the error