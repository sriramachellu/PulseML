FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA wheel.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1" \
 && pip install -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY docs/model_card.md docs/
COPY .streamlit/ .streamlit/
COPY outputs/ outputs/

RUN useradd --create-home --uid 1000 pulseml && chown -R pulseml /app
USER pulseml

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8000/health').status != 200)"

# Defaults to port 8000; honours $PORT on platforms such as Render.
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
