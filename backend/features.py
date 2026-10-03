"""
Extra features for AI Recruitment Intelligence (plug-in module).

Add to backend/main.py, RIGHT AFTER `app = FastAPI(...)` and BEFORE any route or app.mount(...):

    import features
    features.install(app)

Install extra packages:  pip install pyjwt reportlab
Optional .env:           JWT_SECRET=long-random-string   STT_MODEL=whisper-large-v3-turbo
"""
import io
import logging
import os
import re
import sqlite3
import time
import hashlib
import hmac
import secrets
import smtplib
from collections import defaultdict, deque
from email.message import EmailMessage
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from xml.sax.saxutils import escape as xml_escape

import jwt
import requests
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from starlette.concurrency import run_in_threadpool

from llm import ask_json

log = logging.getLogger("recruit")
router = APIRouter(prefix="/api")

DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.db"))
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")


# ───────────────────────── database (SQLite, stdlib only) ─────────────────────────
@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with db() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                pw TEXT NOT NULL, salt TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS interviews(
                id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, role TEXT, score REAL,
                recommendation TEXT, wpm REAL, fillers INTEGER, tab_switches INTEGER,
                report TEXT, created TEXT NOT NULL);
            """
        )


# ───────────────────────── owner email alerts ─────────────────────────
# .env settings (all optional; with no NOTIFY_TO nothing is sent):
#   NOTIFY_TO=you@gmail.com
#   RESEND_API_KEY=re_xxx             (recommended: HTTPS API, works on Render's free tier)
#   NOTIFY_FROM=AI Recruitment <onboarding@resend.dev>
# or SMTP fallback (blocked on many free hosts, fine locally):
#   SMTP_HOST=smtp.gmail.com  SMTP_PORT=465  SMTP_USER=you@gmail.com  SMTP_PASS=<gmail app password>
def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def alert_args(kind: str, email: str, request: Request) -> tuple:
    """Builds (subject, body). Never includes passwords or tokens."""
    ist = timezone(timedelta(hours=5, minutes=30))
    when = datetime.now(ist).strftime("%d %b %Y, %I:%M %p IST")
    body = (f"{kind}\n\nUser:   {email}\nTime:   {when}\nIP:     {client_ip(request)}\n"
            f"Device: {request.headers.get('user-agent', 'unknown')[:150]}\n")
    return f"[AI Recruitment] {kind}: {email}", body


def notify_owner(subject: str, body: str) -> None:
    """Runs in the background and never raises: a mail problem must not break login."""
    to = os.getenv("NOTIFY_TO")
    if not to:
        return
    try:
        key = os.getenv("RESEND_API_KEY")
        if key:
            r = requests.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {key}"},
                json={"from": os.getenv("NOTIFY_FROM", "AI Recruitment <onboarding@resend.dev>"),
                      "to": [to], "subject": subject, "text": body},
                timeout=15)
            if not r.ok:
                log.warning("Resend error %s: %s", r.status_code, r.text[:200])
            return
        host = os.getenv("SMTP_HOST")
        if host:
            msg = EmailMessage()
            msg["Subject"], msg["From"], msg["To"] = subject, os.getenv("SMTP_USER", ""), to
            msg.set_content(body)
            with smtplib.SMTP_SSL(host, int(os.getenv("SMTP_PORT", "465")), timeout=15) as s:
                s.login(os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASS", ""))
                s.send_message(msg)
    except Exception as e:  # noqa: BLE001
        log.warning("Notification failed: %s", e)


# ───────────────────────── authentication (JWT) ─────────────────────────
def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


class AuthReq(BaseModel):
    email: str
    password: str


def _make_token(uid: int, email: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=7)
    return jwt.encode({"sub": str(uid), "email": email, "exp": exp}, JWT_SECRET, algorithm="HS256")


def current_user(request: Request) -> dict:
    h = request.headers.get("Authorization", "")
    if not h.startswith("Bearer "):
        raise HTTPException(401, "Please log in first.")
    try:
        p = jwt.decode(h[7:], JWT_SECRET, algorithms=["HS256"])
        return {"id": int(p["sub"]), "email": p["email"]}
    except Exception:
        raise HTTPException(401, "Session expired. Please log in again.")


@router.post("/auth/register")
def register(b: AuthReq, request: Request, bg: BackgroundTasks):
    email = b.email.strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise HTTPException(400, "Enter a valid email address.")
    if len(b.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    salt = secrets.token_hex(16)
    try:
        with db() as c:
            cur = c.execute(
                "INSERT INTO users(email,pw,salt,created) VALUES(?,?,?,?)",
                (email, _hash(b.password, salt), salt, datetime.now(timezone.utc).isoformat()),
            )
            uid = cur.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(409, "That email is already registered. Try logging in.")
    bg.add_task(notify_owner, *alert_args("New registration", email, request))
    return {"token": _make_token(uid, email), "email": email}


@router.post("/auth/login")
def login(b: AuthReq, request: Request, bg: BackgroundTasks):
    email = b.email.strip().lower()
    with db() as c:
        u = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not u or not hmac.compare_digest(u["pw"], _hash(b.password, u["salt"])):
        raise HTTPException(401, "Wrong email or password.")
    bg.add_task(notify_owner, *alert_args("User login", email, request))
    return {"token": _make_token(u["id"], email), "email": email}


# ───────────────────────── interview history ─────────────────────────
class SaveReq(BaseModel):
    role: str = ""
    score: float
    recommendation: str = ""
    wpm: float | None = None
    fillers: int = 0
    tab_switches: int = 0
    report: dict = {}


@router.post("/history/save")
def history_save(b: SaveReq, request: Request):
    import json

    u = current_user(request)
    with db() as c:
        c.execute(
            "INSERT INTO interviews(user_id,role,score,recommendation,wpm,fillers,tab_switches,report,created)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (u["id"], b.role, b.score, b.recommendation, b.wpm, b.fillers, b.tab_switches,
             json.dumps(b.report), datetime.now(timezone.utc).isoformat()),
        )
    return {"saved": True}


@router.get("/history")
def history(request: Request):
    u = current_user(request)
    with db() as c:
        rows = c.execute(
            "SELECT id,created,role,score,recommendation,wpm,fillers,tab_switches "
            "FROM interviews WHERE user_id=? ORDER BY id", (u["id"],)).fetchall()
    return {"items": [dict(r) for r in rows]}


# ───────────────────────── speech analytics ─────────────────────────
FILLERS = ["um", "umm", "uh", "uhh", "erm", "you know", "basically", "actually",
           "literally", "sort of", "kind of", "i mean"]


class SpeechReq(BaseModel):
    text: str
    seconds: float | None = None


def analyze_speech(text: str, seconds: float | None = None) -> dict:
    low = text.lower()
    words = re.findall(r"[a-z0-9']+", low)
    found = {}
    for f in FILLERS:
        n = len(re.findall(r"\b" + re.escape(f) + r"\b", low))
        if n:
            found[f] = n
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    wpm = round(len(words) / (seconds / 60)) if seconds and seconds >= 3 else None
    pace = None
    if wpm:
        pace = "too fast" if wpm > 170 else "too slow" if wpm < 100 else "good pace"
    return {
        "words": len(words),
        "wpm": wpm,
        "pace": pace,
        "filler_total": sum(found.values()),
        "fillers": found,
        "avg_sentence_words": round(len(words) / len(sentences), 1) if sentences else 0,
    }


@router.post("/speech/analyze")
def speech_analyze(b: SpeechReq):
    return analyze_speech(b.text, b.seconds)


# ───────────────────────── STAR-method coach ─────────────────────────
class StarReq(BaseModel):
    question: str
    answer: str


@router.post("/star/check")
def star_check(b: StarReq):
    return ask_json(
        "You are an interview coach. Judge whether the answer follows the STAR method "
        "(Situation, Task, Action, Result). Be honest, do not inflate.",
        f"Question: {b.question[:1000]}\nAnswer: {b.answer[:3000]}\n\n"
        'Return JSON: {"is_behavioural": true/false, "situation": true/false, "task": true/false, '
        '"action": true/false, "result": true/false, "tip": "one short, specific sentence"}',
        2000,
    )


# ───────────────────────── bulk resume ranking ─────────────────────────
def extract_text(name: str, data: bytes) -> str:
    n = name.lower()
    if n.endswith(".pdf"):
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    if n.endswith(".docx"):
        import docx

        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", "ignore")


def rank_resumes(jd: str, resumes: dict) -> list:
    names = list(resumes)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        m = vec.fit_transform([jd] + [resumes[n] for n in names])
    except ValueError:
        raise HTTPException(400, "Job description or resumes contain no usable text.")
    sims = cosine_similarity(m[0], m[1:])[0]
    terms = vec.get_feature_names_out()
    row = m[0].toarray()[0]
    top = [terms[i] for i in row.argsort()[::-1][:25] if row[i] > 0]
    out = []
    for i, n in enumerate(names):
        low = resumes[n].lower()
        matched = [t for t in top if t in low]
        missing = [t for t in top if t not in low]
        cov = len(matched) / len(top) if top else 0
        score = round(100 * (0.6 * min(1.0, float(sims[i]) * 2) + 0.4 * cov))
        out.append({"name": n, "score": score, "matched": matched[:8], "missing": missing[:8]})
    out.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


@router.post("/bulk-rank")
async def bulk_rank(job_description: str = Form(...), files: list[UploadFile] = File(...)):
    if not job_description.strip():
        raise HTTPException(400, "Add a job description first.")
    if len(files) > 20:
        raise HTTPException(400, "Upload at most 20 resumes at a time.")
    resumes = {}
    for f in files:
        data = await f.read()
        if len(data) > 5 * 1024 * 1024:
            raise HTTPException(413, f"{f.filename} is larger than 5 MB.")
        try:
            text = extract_text(f.filename or "resume.txt", data)
        except Exception as e:
            raise HTTPException(400, f"Could not read {f.filename}: {e}")
        if text.strip():
            resumes[f.filename] = text
    if not resumes:
        raise HTTPException(400, "No readable text found in the uploaded files.")
    return {"results": await run_in_threadpool(rank_resumes, job_description, resumes)}


# ───────────────────────── AI resume rewriter ─────────────────────────
class RewriteReq(BaseModel):
    resume: str
    job_description: str


@router.post("/rewrite-resume")
def rewrite_resume(b: RewriteReq):
    if not b.resume.strip() or not b.job_description.strip():
        raise HTTPException(400, "Add a resume and a job description first.")
    return ask_json(
        "You are an expert resume writer. NEVER invent facts, employers, tools or numbers. "
        "Only rephrase what is already in the resume; where a metric would help, write a "
        "placeholder like [add real number] for the candidate to fill in.",
        f"JOB DESCRIPTION:\n{b.job_description[:4000]}\n\nRESUME:\n{b.resume[:6000]}\n\n"
        'Return JSON: {"summary": "2-3 sentence professional summary", '
        '"improvements": [{"original": "...", "improved": "...", "why": "..."}], '
        '"keywords_to_add": ["..."]} with 5 to 8 improvements.',
        3500,
    )


# ───────────────────────── speech-to-text (Groq Whisper) ─────────────────────────
@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty recording.")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "Recording too large (max 20 MB).")

    def call():
        return requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            files={"file": (file.filename or "answer.webm", data, file.content_type or "audio/webm")},
            data={"model": os.getenv("STT_MODEL", "whisper-large-v3-turbo"),
                  "response_format": "json", "language": os.getenv("STT_LANG", "en")},
            timeout=60,
        )

    r = await run_in_threadpool(call)
    if not r.ok:
        raise HTTPException(502, f"Whisper error {r.status_code}: {r.text[:200]}")
    return {"text": r.json().get("text", "").strip()}


# ───────────────────────── PDF report ─────────────────────────
class PdfReq(BaseModel):
    role: str = ""
    overall: float = 0
    recommendation: str = ""
    strengths: list = []
    weaknesses: list = []
    next_steps: list = []
    wpm: float | None = None
    fillers: int = 0
    tab_switches: int = 0
    transcript: list = []


def build_pdf(d: PdfReq) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    x = lambda v: xml_escape(str(v))
    s = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Interview Report")
    E = [
        Paragraph("Interview Report", s["Title"]),
        Paragraph(f"Role: {x(d.role or '-')} &nbsp;|&nbsp; Score: <b>{d.overall}/100</b> "
                  f"&nbsp;|&nbsp; Recommendation: <b>{x(d.recommendation)}</b>", s["Normal"]),
        Spacer(1, 8),
        Paragraph(f"Delivery: {d.wpm if d.wpm else 'n/a'} words/min, {d.fillers} filler words, "
                  f"{d.tab_switches} tab switches", s["Normal"]),
        Spacer(1, 12),
    ]
    for title, items in (("Strengths", d.strengths), ("Weaknesses", d.weaknesses), ("Next steps", d.next_steps)):
        E.append(Paragraph(title, s["Heading2"]))
        E += [Paragraph("• " + x(i), s["Normal"]) for i in items] or [Paragraph("-", s["Normal"])]
    if d.transcript:
        E += [Spacer(1, 12), Paragraph("Transcript", s["Heading2"])]
        for i, t in enumerate(d.transcript, 1):
            E += [Paragraph(f"<b>Q{i}:</b> {x(t.get('q', ''))}", s["Normal"]),
                  Paragraph(f"<b>Answer:</b> {x(t.get('a', ''))}", s["Normal"]),
                  Paragraph(f"<i>Feedback:</i> {x(t.get('feedback', ''))}", s["Normal"]),
                  Spacer(1, 8)]
    doc.build(E)
    return buf.getvalue()


@router.post("/report/pdf")
def report_pdf(b: PdfReq):
    pdf = build_pdf(b)
    return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf",
                             headers={"Content-Disposition": 'attachment; filename="interview-report.pdf"'})


# ───────────────────────── install: router + rate limit + logging ─────────────────────────
HITS = defaultdict(deque)


def install(app):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db()
    app.include_router(router)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            ip = request.client.host if request.client else "unknown"
            now, q = time.time(), HITS[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= int(os.getenv("RATE_LIMIT_PER_MIN", "60")):
                return JSONResponse({"detail": "Too many requests. Please slow down."}, status_code=429)
            q.append(now)
        t0 = time.time()
        resp = await call_next(request)
        log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, resp.status_code, (time.time() - t0) * 1000)
        return resp