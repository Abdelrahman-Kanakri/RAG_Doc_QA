FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# uv tuning for containers
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 1) Deps layer — copy ONLY lock + manifest so this layer caches
#    until dependencies actually change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

# 2) Source layer — changes often, cheap to rebuild
COPY app/ ./app/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

# Put the project venv on PATH so `uvicorn` resolves
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]