FROM python:3.10-slim

WORKDIR /app

COPY requirements-render.txt .

RUN pip install --no-cache-dir --default-timeout=1000 \
    torch==1.13.1+cpu \
    torchvision==0.14.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --default-timeout=1000 \
    -r requirements-render.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]