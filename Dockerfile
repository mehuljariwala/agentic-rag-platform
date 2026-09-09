# syntax=docker/dockerfile:1

# ---- build ------------------------------------------------------------------
FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Copy only what the wheel build needs so this layer caches across source edits.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir build hatchling \
 && python -m build --wheel --outdir /dist

# ---- runtime ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAGKIT_PROVIDER=mock

# Non-root. The app writes nothing to disk, so no ownership juggling is needed.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 ragkit

WORKDIR /app

COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl "fastapi>=0.110" "uvicorn[standard]>=0.27" \
 && rm -f /tmp/*.whl

# Eval harness ships in the image so the regression gate can run in CI or as a
# one-off container without a second build.
COPY evals ./evals

USER ragkit

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "ragkit.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
