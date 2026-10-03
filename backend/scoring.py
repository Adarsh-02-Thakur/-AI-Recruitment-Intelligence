"""Transparent, explainable scoring engine (no LLM needed)."""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# skill -> aliases. Extend this taxonomy (or load from ESCO / O*NET datasets).
SKILLS = {
    "python": ["python"], "java": ["java"], "javascript": ["javascript", "js", "node.js", "nodejs"],
    "typescript": ["typescript"], "c++": ["c++"], "sql": ["sql", "mysql", "postgresql", "postgres"],
    "mongodb": ["mongodb", "nosql"], "react": ["react", "react.js", "reactjs"], "angular": ["angular"],
    "django": ["django"], "flask": ["flask"], "fastapi": ["fastapi"], "spring boot": ["spring boot", "spring"],
    "html/css": ["html", "css"], "rest api": ["rest", "restful", "api"], "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"], "aws": ["aws", "amazon web services"], "azure": ["azure"],
    "gcp": ["gcp", "google cloud"], "git": ["git", "github", "gitlab"], "ci/cd": ["ci/cd", "jenkins", "github actions"],
    "linux": ["linux"], "machine learning": ["machine learning", "ml"], "deep learning": ["deep learning", "neural network"],
    "nlp": ["nlp", "natural language processing"], "pandas": ["pandas"], "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"], "tensorflow": ["tensorflow"], "pytorch": ["pytorch"],
    "data analysis": ["data analysis", "data analytics"], "power bi": ["power bi", "powerbi"], "tableau": ["tableau"],
    "excel": ["excel"], "statistics": ["statistics", "statistical"], "agile": ["agile", "scrum"],
    "communication": ["communication"], "leadership": ["leadership", "team lead"], "problem solving": ["problem solving", "problem-solving"],
    "system design": ["system design"], "microservices": ["microservices"], "data structures": ["data structures", "algorithms"],
    "testing": ["unit testing", "pytest", "junit", "selenium", "testing"], "devops": ["devops"],
    "llm": ["llm", "large language model", "generative ai", "langchain"], "figma": ["figma"],
}
SECTIONS = {"summary": r"\b(summary|objective|profile)\b", "experience": r"\b(experience|employment|work history)\b",
            "education": r"\beducation\b", "skills": r"\bskills\b", "projects": r"\bprojects?\b",
            "certifications": r"\b(certifications?|courses)\b"}
ACTION_VERBS = r"\b(built|developed|designed|led|created|improved|reduced|increased|launched|implemented|optimized|automated|managed|delivered)\b"


def _has(text, alias):
    return re.search(r"(?<![\w+#])" + re.escape(alias) + r"(?![\w+#])", text) is not None


def extract_skills(text: str) -> set:
    t = text.lower()
    return {skill for skill, aliases in SKILLS.items() if any(_has(t, a) for a in aliases)}


def years_of_experience(text: str) -> float:
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", text.lower())]
    return max(nums) if nums else 0.0


def skill_gap(resume: str, jd: str) -> dict:
    r, j = extract_skills(resume), extract_skills(jd)
    matched, missing = sorted(r & j), sorted(j - r)
    return {"required": sorted(j), "matched": matched, "missing": missing,
            "extra": sorted(r - j), "coverage": round(len(matched) / len(j) * 100, 1) if j else 0.0}


def semantic_similarity(a: str, b: str) -> float:
    try:
        m = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([a, b])
        return float(cosine_similarity(m[0], m[1])[0][0])
    except ValueError:
        return 0.0


def ats_score(resume: str, jd: str) -> dict:
    gap = skill_gap(resume, jd)
    low = resume.lower()
    keyword = gap["coverage"] / 100
    sim = min(semantic_similarity(resume, jd) * 2.2, 1.0)  # TF-IDF cosine is naturally low; rescale
    found = [s for s, p in SECTIONS.items() if re.search(p, low)]
    contact = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", resume)) + bool(re.search(r"\+?\d[\d\s\-]{8,}", resume))
    structure = (len(found) / len(SECTIONS)) * 0.8 + (contact / 2) * 0.2
    bullets_with_numbers = len(re.findall(r"\d+\s*%|\$\s*\d+|\b\d{2,}\b", resume))
    verbs = len(re.findall(ACTION_VERBS, low))
    impact = min(bullets_with_numbers / 6, 1) * 0.5 + min(verbs / 8, 1) * 0.5
    need, have = years_of_experience(jd), years_of_experience(resume)
    exp = 1.0 if need == 0 else min(have / need, 1.0)
    words = len(resume.split())
    length_ok = 1.0 if 250 <= words <= 1000 else 0.6
    parts = {"keyword_match": keyword * 40, "semantic_relevance": sim * 20, "structure": structure * 15,
             "impact_and_verbs": impact * 15, "experience_fit": exp * 5, "length": length_ok * 5}
    tips = []
    if gap["missing"]: tips.append("Add (truthfully) these missing keywords: " + ", ".join(gap["missing"][:8]))
    missing_sec = [s for s in SECTIONS if s not in found]
    if missing_sec: tips.append("Add sections: " + ", ".join(missing_sec))
    if bullets_with_numbers < 4: tips.append("Quantify achievements (%, $, users, time saved).")
    if verbs < 5: tips.append("Start bullets with strong action verbs.")
    if length_ok < 1: tips.append("Aim for roughly 1-2 pages (250-1000 words).")
    return {"score": round(sum(parts.values()), 1), "breakdown": {k: round(v, 1) for k, v in parts.items()},
            "sections_found": found, "tips": tips, "skill_gap": gap}


def job_match(resume: str, jobs: list) -> list:
    out = []
    for job in jobs:
        gap = skill_gap(resume, job["description"])
        sim = semantic_similarity(resume, job["description"])
        match = gap["coverage"] * 0.7 + min(sim * 2.2, 1) * 100 * 0.3
        out.append({"id": job["id"], "title": job["title"], "match": round(match, 1),
                    "matched": gap["matched"], "missing": gap["missing"]})
    return sorted(out, key=lambda x: -x["match"])


def career_readiness(ats: float, match: float, interview: float | None, resume: str) -> dict:
    low = resume.lower()
    portfolio = 100 if re.search(r"github\.com|portfolio|linkedin\.com", low) else 40
    proof = 100 if re.search(r"\bprojects?\b", low) and re.search(r"certif", low) else 60 if re.search(r"\bprojects?\b", low) else 30
    dims = {"Resume (ATS)": ats, "Job Match": match, "Proof of Work": proof, "Online Presence": portfolio}
    if interview is not None: dims["Interview"] = interview
    overall = sum(dims.values()) / len(dims)
    level = "Job Ready" if overall >= 75 else "Almost There" if overall >= 55 else "Needs Work"
    return {"overall": round(overall, 1), "level": level, "dimensions": {k: round(v, 1) for k, v in dims.items()}}
