#!/bin/sh
# Container start: wait for ChromaDB, prepare the database, then serve with gunicorn.
set -e

if [ "${VECTOR_STORE:-chroma}" = "chroma" ]; then
  python - <<'PY'
import os, sys, time, urllib.request
url = os.environ.get("CHROMA_URL", "http://chroma:8000").rstrip("/") + "/api/v2/heartbeat"
for attempt in range(60):
    try:
        urllib.request.urlopen(url, timeout=2)
        print("ChromaDB is up", flush=True)
        sys.exit(0)
    except Exception:
        time.sleep(1)
print(f"ChromaDB did not answer at {url}; starting anyway (search uses the in-memory vectors)", flush=True)
PY
fi

python manage.py migrate --noinput -v 0

# Threads, not just processes: an answer can wait up to 90 s on the model, and a thread waiting on the
# network does not block other visitors. The timeout is above the 90 s answer deadline.
exec gunicorn webapp.wsgi:application \
  --bind 0.0.0.0:8000 \
  --worker-class gthread --workers "${GUNICORN_WORKERS:-2}" --threads "${GUNICORN_THREADS:-8}" \
  --timeout 150 --graceful-timeout 30 --keep-alive 5 \
  --access-logfile - --error-logfile -
