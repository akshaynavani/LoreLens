FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY data ./data

RUN useradd --create-home lorelens && mkdir -p /app/.cache && chown -R lorelens /app
USER lorelens

EXPOSE 8000
CMD ["uvicorn", "lorelens.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
