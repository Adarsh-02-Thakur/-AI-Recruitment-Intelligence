FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
WORKDIR /app/backend
ENV PORT=8000
EXPOSE 8000
# Secrets (GROQ_API_KEY, JWT_SECRET, ...) are passed at runtime, never baked into the image:
#   docker run -p 8000:8000 --env-file .env ai-recruitment
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]