"""Accounts: sign up, log in, forgot/reset password. Stdlib only (no new packages). Data in backend/users.db"""
import base64, hashlib, hmac, json, os, re, secrets, sqlite3, time
import requests
from dotenv import load_dotenv
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from pydantic import BaseModel

load_dotenv()
router = APIRouter(prefix="/api/account")
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")
SECRET = (os.getenv("JWT_SECRET") or "dev-only-change-me").encode()
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ATTEMPTS: dict = {}


def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS users(email TEXT PRIMARY KEY, name TEXT, pw TEXT, created REAL,
                 reset_hash TEXT, reset_exp REAL, reset_tries INTEGER DEFAULT 0)""")
    return c


def hash_pw(pw: str) -> str:
    salt = os.urandom(16)
    return salt.hex() + ":" + hashlib.scrypt(pw.encode(), salt=salt, n=2**14, r=8, p=1).hex()


def check_pw(pw: str, stored: str) -> bool:
    salt, h = stored.split(":")
    return hmac.compare_digest(hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1).hex(), h)


def make_token(email: str, days: int = 7) -> str:
    body = base64.urlsafe_b64encode(json.dumps({"sub": email, "exp": time.time() + days * 86400}).encode()).decode()
    return body + "." + hmac.new(SECRET, body.encode(), hashlib.sha256).hexdigest()


def read_token(tok: str):
    try:
        body, sig = tok.split(".")
        if not hmac.compare_digest(hmac.new(SECRET, body.encode(), hashlib.sha256).hexdigest(), sig): return None
        data = json.loads(base64.urlsafe_b64decode(body))
        return data["sub"] if data["exp"] > time.time() else None
    except Exception:
        return None


def current_user(authorization: str = Header(None)):
    """Use as a dependency on any route you want to protect: Depends(current_user)"""
    email = read_token((authorization or "").replace("Bearer ", ""))
    if not email: raise HTTPException(401, "Please log in")
    return email


class SignupIn(BaseModel): name: str; email: str; password: str
class LoginIn(BaseModel): email: str; password: str
class ForgotIn(BaseModel): email: str
class ResetIn(BaseModel): email: str; code: str; password: str


def _check(email, password=None):
    if not EMAIL_RE.match(email.strip()): raise HTTPException(422, "Enter a valid email address")
    if password is not None and len(password) < 8: raise HTTPException(422, "Password must be at least 8 characters")


def _session(row):
    return {"token": make_token(row["email"]), "user": {"name": row["name"], "email": row["email"]}}


@router.post("/signup")
def signup(b: SignupIn, bg: BackgroundTasks):
    email = b.email.strip().lower(); _check(email, b.password)
    if not b.name.strip(): raise HTTPException(422, "Enter your name")
    with db() as c:
        if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone(): raise HTTPException(409, "An account with this email already exists")
        c.execute("INSERT INTO users(email,name,pw,created) VALUES(?,?,?,?)", (email, b.name.strip()[:60], hash_pw(b.password), time.time()))
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    bg.add_task(notify_signup, email, row["name"])
    return _session(row)


@router.post("/login")
def login(b: LoginIn, request: Request, bg: BackgroundTasks):
    email = b.email.strip().lower()
    n, t = ATTEMPTS.get(email, (0, time.time()))
    if time.time() - t > 300: n, t = 0, time.time()
    if n >= 5: raise HTTPException(429, "Too many attempts. Try again in a few minutes.")
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row or not check_pw(b.password, row["pw"]):
        ATTEMPTS[email] = (n + 1, t); raise HTTPException(401, "Invalid email or password")
    ATTEMPTS.pop(email, None)
    bg.add_task(notify_login, email, row["name"], request.client.host if request.client else "unknown")
    return _session(row)


FROM = os.getenv("RESEND_FROM", "RecruitAI <onboarding@resend.dev>")


def _wrap(title: str, body: str) -> str:
    return (f"<div style='font-family:Arial,sans-serif;max-width:520px;margin:auto;border-radius:16px;overflow:hidden;border:1px solid #e5e3ff'>"
            f"<div style='background:linear-gradient(135deg,#7c5cff,#ff5ca8);color:#fff;padding:22px;font-size:20px;font-weight:bold'>🧠 RecruitAI</div>"
            f"<div style='padding:22px;color:#222'><h2 style='margin-top:0'>{title}</h2>{body}</div></div>")


def send_email(to: str, subject: str, html: str) -> bool:
    key = os.getenv("RESEND_API_KEY")
    if not key:
        print("[MAIL] RESEND_API_KEY is missing in .env, so no email was sent"); return False
    try:
        r = requests.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {key}"}, timeout=20,
                          json={"from": FROM, "to": [to], "subject": subject, "html": html})
        if r.status_code >= 300:
            print(f"[MAIL] FAILED to {to}: {r.status_code} {r.text}"); return False
        print(f"[MAIL] Sent '{subject}' to {to}"); return True
    except Exception as e:
        print(f"[MAIL] FAILED to {to}: {e}"); return False


def _admin(subject: str, html: str):
    to = os.getenv("NOTIFY_TO")
    if to: send_email(to, subject, _wrap(subject, html))


def notify_signup(email: str, name: str):
    send_email(email, "Welcome to RecruitAI 🎉", _wrap(f"Welcome, {name}!", "<p>Your account is ready. Analyze your resume, practice AI interviews and track your career readiness.</p>"))
    _admin("New signup", f"<p><b>{name}</b> ({email}) just created an account.</p>")


def notify_login(email: str, name: str, ip: str):
    when = datetime.now().strftime("%d %b %Y, %I:%M %p")
    send_email(email, "New login to your RecruitAI account", _wrap("New login detected",
               f"<p>Hi {name}, your account was just logged in on <b>{when}</b> (IP: {ip}).</p><p>If this wasn't you, reset your password immediately.</p>"))
    _admin("User login", f"<p><b>{name}</b> ({email}) logged in at {when} from {ip}.</p>")


def _send_code(email: str, code: str):
    ok = send_email(email, "Your RecruitAI password reset code", _wrap("Password reset",
         f"<p>Your reset code is</p><p style='font-size:30px;letter-spacing:6px;font-weight:bold'>{code}</p><p>It expires in 15 minutes. If you didn't ask for this, ignore this email.</p>"))
    if not ok:  # dev fallback: show the code in the server terminal only
        print(f"[DEV] Reset code for {email}: {code}")


@router.post("/forgot")
def forgot(b: ForgotIn):
    email = b.email.strip().lower(); _check(email)
    with db() as c:
        if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            code = f"{secrets.randbelow(10**6):06d}"
            c.execute("UPDATE users SET reset_hash=?, reset_exp=?, reset_tries=0 WHERE email=?",
                      (hashlib.sha256(code.encode()).hexdigest(), time.time() + 900, email))
            _send_code(email, code)
    return {"message": "If that email has an account, a 6-digit reset code has been sent."}  # same reply either way


@router.post("/reset")
def reset(b: ResetIn):
    email = b.email.strip().lower(); _check(email, b.password)
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        bad = HTTPException(400, "Invalid or expired code")
        if not row or not row["reset_hash"] or row["reset_exp"] < time.time() or row["reset_tries"] >= 5: raise bad
        if not hmac.compare_digest(hashlib.sha256(b.code.strip().encode()).hexdigest(), row["reset_hash"]):
            c.execute("UPDATE users SET reset_tries=reset_tries+1 WHERE email=?", (email,)); c.commit(); raise bad
        c.execute("UPDATE users SET pw=?, reset_hash=NULL, reset_exp=NULL WHERE email=?", (hash_pw(b.password), email))
    return {"message": "Password updated. You can log in now."}


@router.get("/me")
def me(authorization: str = Header(None)):
    email = current_user(authorization)
    with db() as c:
        row = c.execute("SELECT name,email FROM users WHERE email=?", (email,)).fetchone()
    if not row: raise HTTPException(401, "Please log in")
    return {"name": row["name"], "email": row["email"]}