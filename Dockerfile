# VoltVision AI REST API.
#
# The image contains code and the public ImageNet ResNet18 weights only. It does
# NOT contain the MVTec AD dataset (CC BY-NC-SA 4.0) or the reference bank:
# mount both at run time (see docker-compose.yml). The bank is validated against
# the dataset's contents on first use, so both mounts are required.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch

WORKDIR /app

# CPU-only wheels keep the image small; no GPU is used.
COPY requirements-api.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-api.txt

# Bake in the pretrained weights so the container starts without internet access.
RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"

COPY config.py ./
COPY src ./src
COPY api ./api

# Run unprivileged. artifacts/ must be writable for the inspection event log.
RUN useradd --create-home --uid 10001 voltvision \
 && mkdir -p /app/artifacts /app/data \
 && chown -R voltvision /app/artifacts
USER voltvision

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# One worker: the engine holds the reference bank in memory and the event log is a CSV file.
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
