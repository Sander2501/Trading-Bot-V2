FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY forex_bot ./forex_bot

RUN pip install --no-cache-dir .

RUN mkdir -p /app/data /app/logs

ENTRYPOINT ["forex-bot"]
