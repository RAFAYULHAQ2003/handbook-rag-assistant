# Handbook Q&A assistant: Django + gunicorn. Vectors live in the ChromaDB service (docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Download the embedding model and embed the handbook once, at build time, so the container starts in
# about a second and never needs the internet for it (HF_HUB_OFFLINE below). Static files are collected
# for WhiteNoise. Both steps need no secrets, so a throwaway key is used just for them.
RUN VECTOR_STORE=memory python -c "from pathlib import Path; from rag.index import HandbookIndex; \
i = HandbookIndex.from_corpus(Path('data/corpus')); print(len(i.passages), 'passages,', i.mode)" \
 && DJANGO_SECRET_KEY=build-only python manage.py collectstatic --noinput -v 0

# Run as an unprivileged user; it owns only what it must write: answer cache, uploads, database.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/.cache/llm /app/media/uploads /app/var \
 && chown -R app:app /app/.cache /app/media /app/var
USER app

ENV HF_HUB_OFFLINE=1 \
    DJANGO_DB_PATH=/app/var/db.sqlite3 \
    UPLOAD_DIR=/app/media/uploads

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

ENTRYPOINT ["sh", "docker/entrypoint.sh"]
