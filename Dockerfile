FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt

COPY --chown=user ./data /app/data
COPY --chown=user ./jitter /app/jitter
COPY --chown=user ./static /app/static

USER user

CMD ["uvicorn", "jitter:app", "--host", "0.0.0.0", "--port", "7860"]
