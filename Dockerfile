# Handbook Q&A assistant: Django + gunicorn. Vectors live in the ChromaDB service (docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# An unprivileged user owns the app files from the start (no `chown -R` layer that would copy the
# embedding model a second time and make the image bigger).
RUN useradd --create-home --uid 1000 app && chown app:app /app
COPY --chown=app:app . .
USER app
RUN mkdir -p /app/.cache/llm /app/media/uploads /app/var

# Download the embedding model and embed the handbook once, at build time, so the container starts in
# about a second and never needs the internet for it (HF_HUB_OFFLINE below). Static files are collected
# for WhiteNoise. Both steps need no secrets, so a throwaway key is used just for them.
RUN VECTOR_STORE=memory python -c "from pathlib import Path; from rag.index import HandbookIndex; \
i = HandbookIndex.from_corpus(Path('data/corpus')); print(len(i.passages), 'passages,', i.mode)" \
 && DJANGO_SECRET_KEY=build-only python manage.py collectstatic --noinput -v 0

ENV HF_HUB_OFFLINE=1 \
    DJANGO_DB_PATH=/app/var/db.sqlite3 \
    UPLOAD_DIR=/app/media/uploads

# the commit this image was built from (set by CI), shown by /api/health so a deploy can be verified
ARG APP_VERSION=dev
ENV APP_VERSION=$APP_VERSION

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

ENTRYPOINT ["sh", "docker/entrypoint.sh"]
