# AI Recruitment Intelligence

## Run
```bash
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY
cd backend && uvicorn main:app --reload
# open http://localhost:8000
```
## Features
- Resume upload (PDF/DOCX/TXT) + parsing
- Explainable ATS score (keywords 40, relevance 20, structure 15, impact 15, experience 5, length 5)
- Skill-gap analysis + AI 4-week learning plan
- Job matching (ranked) · Career-readiness radar
- AI interviewer: adaptive questions, per-answer scoring, voice in/out, final hiring report
