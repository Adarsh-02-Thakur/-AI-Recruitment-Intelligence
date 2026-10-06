import io, uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pdfplumber, docx
import scoring
from llm import ask_json
from auth import router as auth_router
from ats_analyzer import router as ats_router
from fastapi.responses import JSONResponse
app = FastAPI(title="AI Recruitment Intelligence")
app.include_router(auth_router)@app.exception_handler(Exception)
async def show_errors(request, exc):
    print(f"[ERROR] {request.url.path}: {type(exc).__name__}: {exc}")
    return JSONResponse({"detail": f"{type(exc).__name__}: {str(exc)[:300]}"}, status_code=500)
app.include_router(ats_router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS = [  # replace with a database / job-board API (Adzuna, JSearch, etc.)
    {"id": 1, "title": "Python Backend Developer", "description": "Python FastAPI Django REST API SQL PostgreSQL Docker AWS Git CI/CD microservices testing 2 years"},
    {"id": 2, "title": "Data Scientist", "description": "Python pandas numpy scikit-learn machine learning statistics SQL data analysis tableau deep learning 3 years"},
    {"id": 3, "title": "Frontend Developer", "description": "JavaScript TypeScript React HTML CSS REST API Git figma testing agile 2 years"},
    {"id": 4, "title": "DevOps Engineer", "description": "Linux Docker Kubernetes AWS CI/CD git python devops terraform monitoring 3 years"},
    {"id": 5, "title": "AI/LLM Engineer", "description": "Python LLM NLP machine learning pytorch FastAPI docker system design deep learning"},
]
SESSIONS: dict = {}
MAX_Q = 5


def extract_text(name: str, data: bytes) -> str:
    n = name.lower()
    if n.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    if n.endswith(".docx"):
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")


@app.post("/api/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    text = extract_text(file.filename, await file.read())
    if len(text.strip()) < 50:
        raise HTTPException(422, "Could not read text from this file (scanned PDF?).")
    return {"text": text}


class AnalyzeIn(BaseModel):
    resume: str
    job_description: str


@app.post("/api/analyze")
def analyze(body: AnalyzeIn):
    ats = scoring.ats_score(body.resume, body.job_description)
    matches = scoring.job_match(body.resume, JOBS)
    ready = scoring.career_readiness(ats["score"], ats["skill_gap"]["coverage"], None, body.resume)
    return {"ats": ats, "job_matches": matches, "readiness": ready}


@app.post("/api/learning-plan")
def learning_plan(body: AnalyzeIn):
    gap = scoring.skill_gap(body.resume, body.job_description)
    return ask_json("You are a career coach.", f"Missing skills: {gap['missing']}. Create a 4-week learning plan as "
                    '{"weeks":[{"week":1,"focus":"","tasks":[""],"free_resources":[""]}]}')


class StartIn(BaseModel):
    resume: str
    job_description: str
    role: str = "the role"
    difficulty: str = "medium"
    mode: str = "real"


@app.post("/api/interview/start")
def start(body: StartIn):
    sid = str(uuid.uuid4())
    gap = scoring.skill_gap(body.resume, body.job_description)
    S = {"ctx": body, "gap": gap, "turns": [], "scores": []}
    SESSIONS[sid] = S
    q = ask_json("You are a professional interviewer.", f"Role: {body.role}. Difficulty: {body.difficulty}. "
                 f"Candidate resume:\n{body.resume[:3000]}\nJob description:\n{body.job_description[:2000]}\n"
                 f"Skills to probe (gaps): {gap['missing'][:5]}. Ask a warm opening question about their background. "
                 'Return {"question": ""}')
    S["current"] = q["question"]
    return {"session_id": sid, "question": q["question"], "number": 1, "total": MAX_Q}


class AnswerIn(BaseModel):
    session_id: str
    answer: str


@app.post("/api/interview/answer")
def answer(body: AnswerIn):
    S = SESSIONS.get(body.session_id)
    if not S:
        raise HTTPException(404, "Session not found")
    n = len(S["turns"]) + 1
    last = n >= MAX_Q
    history = "\n".join(f"Q: {t['q']}\nA: {t['a']}" for t in S["turns"])
    res = ask_json(
        "You are a strict but fair interviewer. Score honestly; do not inflate.",
        f"Role: {S['ctx'].role}\nJD:\n{S['ctx'].job_description[:1500]}\nPrevious:\n{history}\n\n"
        f"Current question: {S['current']}\nCandidate answer: {body.answer}\n\n"
        "Score 0-10 for relevance, technical_depth, communication, structure (STAR for behavioral). "
        f"{'No next question; this was the last.' if last else 'Then ask ONE follow-up or new question (mix technical/behavioral, difficulty ' + S['ctx'].difficulty + ').'}\n"
        'Return {"scores":{"relevance":0,"technical_depth":0,"communication":0,"structure":0},'
        '"feedback":"2 sentences","ideal_answer_hint":"1 sentence","next_question":""}')
    S["turns"].append({"q": S["current"], "a": body.answer, **res})
    S["scores"].append(res["scores"])
    S["current"] = res.get("next_question", "")
    return {"feedback": res["feedback"], "scores": res["scores"], "hint": res.get("ideal_answer_hint"),
            "next_question": None if last else S["current"], "number": n + 1, "total": MAX_Q, "finished": last}


@app.get("/api/interview/report/{sid}")
def report(sid: str):
    S = SESSIONS.get(sid)
    if not S or not S["scores"]:
        raise HTTPException(404, "No completed answers")
    keys = S["scores"][0].keys()
    avg = {k: round(sum(s[k] for s in S["scores"]) / len(S["scores"]), 1) for k in keys}
    overall = round(sum(avg.values()) / len(avg) * 10, 1)
    summary = ask_json("You are a hiring manager.", f"Transcript: {[(t['q'], t['a']) for t in S['turns']]}\nAverages: {avg}. "
                       'Return {"strengths":[""],"weaknesses":[""],"recommendation":"Strong Hire|Hire|Borderline|No Hire","next_steps":[""]}')
    return {"overall": overall, "averages": avg, **summary}


class ReadyIn(BaseModel):
    ats: float
    match: float
    interview: float | None = None
    resume: str


@app.post("/api/readiness")
def readiness(b: ReadyIn):
    return scoring.career_readiness(b.ats, b.match, b.interview, b.resume)


class CoachIn(BaseModel):
    session_id: str
    message: str = ""


@app.post("/api/interview/coach")
def coach(b: CoachIn):
    S = SESSIONS.get(b.session_id)
    if not S:
        raise HTTPException(404, "Session not found")
    if S["ctx"].mode != "practice":
        raise HTTPException(403, "AI coach is only available in Practice mode")
    if b.message.strip():
        task = f'The candidate asks: "{b.message}". Answer helpfully and briefly, relating it to the interview.'
    else:
        task = "The candidate does not know the answer. Give a model answer (use STAR for behavioral questions), 120-170 words, then 2 key takeaways."
    return ask_json("You are a supportive interview coach.",
                    f"Role: {S['ctx'].role}\nResume:\n{S['ctx'].resume[:2500]}\nCurrent interview question: {S['current']}\n{task}\n"
                    'Return {"reply": ""}')


class ChatIn(BaseModel):
    messages: list[dict]
    use_context: bool = False
    resume: str = ""
    job_description: str = ""


@app.post("/api/chat")
def chat_api(b: ChatIn):
    from chat_llm import chat
    msgs = [{"role": m["role"], "content": str(m.get("content", ""))[:4000]}
            for m in b.messages[-12:] if m.get("role") in ("user", "assistant")]
    if not msgs or msgs[-1]["role"] != "user":
        raise HTTPException(400, "Last message must be from the user")
    system = ("You are a knowledgeable, honest assistant for a recruitment and career platform, but you can answer any question. "
              "Give clear, accurate, well-structured answers. If you are not sure or the topic may have changed recently, say so "
              "instead of guessing. Never invent facts, numbers or sources. Use code blocks for code.")
    if b.use_context and b.resume.strip():
        system += f"\n\nCandidate resume:\n{b.resume[:3000]}\n\nTarget job description:\n{b.job_description[:2000]}"
    return {"reply": chat(system, msgs)}


# Must stay LAST: serves the website from the frontend folder
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")