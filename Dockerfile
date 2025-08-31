FROM python:3.12.11-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --root-user-action=ignore --no-cache-dir --upgrade pip
RUN pip install --root-user-action=ignore --no-cache-dir -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/cpu

FROM python:3.12.11-slim-bookworm AS runner

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt update && apt install espeak -y && rm -rf /var/lib/apt/lists/* && apt clean && apt autoremove -y

WORKDIR /app
RUN useradd --system araceebia && mkdir /home/araceebia && chown araceebia:araceebia /home/araceebia
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --chown=araceebia:araceebia . .

WORKDIR /app/src
USER araceebia
EXPOSE 8000
CMD ["python", "-m", "gunicorn", "api:app", "--worker-class", "uvicorn_worker.UvicornWorker", "--bind", "0.0.0.0:8000"]