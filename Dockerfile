# Debate-GPT backend — multi-stage build (PRD §8.1)
#
# Stage 1 (builder): compiles wheels for every dependency, including
# psycopg2-binary (used only by Alembic migrations, sync driver) and
# asyncpg (the app's real async driver). Discarded after stage 2.
#
# Stage 2 (runtime): installs only the pre-built wheels — no compilers,
# no build tools, no pip cache — then copies the application code.
# Runs as a non-root user.
#
# Entry point: run_api.py (the Day 3 launcher). It inserts `src/` onto
# sys.path and calls debate_gpt.__main__:main(), which starts uvicorn
# against the `debate_gpt.api:create_app` factory. Railway/Render inject
# a $PORT env var at runtime — the CMD below respects it via the shell
# form, falling back to 8000 for local `docker run` testing.

# ---------- Stage 1: builder ----------
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim

WORKDIR /app

# Install from the pre-built wheels only — no network access needed here,
# no build toolchain in the final image.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels /wheels/*.whl \
    && rm -rf /wheels

# Application code. See .dockerignore for what's excluded (tests/,
# frontend/, .git, .env, venv/, logs/, node_modules/, etc.) — the
# frontend is deployed separately to Vercel, not part of this image.
COPY src/ ./src/
COPY migrations/ ./migrations/
COPY alembic.ini .
COPY run_api.py .

# Pre-create the logs/ directory and hand ownership to the non-root user
# BEFORE switching to it. observability/logging.py calls
# os.makedirs("logs", exist_ok=True) at import time to set up the JSON
# file sink — without this, /app is root-owned and appuser gets a
# PermissionError on startup (the exact crash this fixes).
RUN mkdir -p /app/logs

# Non-root user for security.
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

# Shell form so ${PORT:-8000} expands — Railway/Render set $PORT at
# runtime; falls back to 8000 for local `docker run -p 8000:8000`.
CMD ["sh", "-c", "python run_api.py --host 0.0.0.0 --port ${PORT:-8000}"]