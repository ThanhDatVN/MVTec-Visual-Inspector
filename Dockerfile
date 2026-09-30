# CPU inference image for the inspection service (inspector serve).
#
# Build:  docker build -t mvi-inspector .
# Export: inspector export -c configs/data/visa_pcb1.yaml \
#             -c configs/recipes/confirmation-v1-patchcore-640.yaml \
#             --data-root <VisA> --category pcb1 --out artifacts/pcb1
# Run:    docker run --rm -p 8000:8000 -v "$PWD/artifacts/pcb1:/artifact:ro" mvi-inspector
#
# No dataset image and no fitted bank is baked in: the artifact (memory bank + meta.json with the
# frozen threshold) is mounted read-only at run time.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch

WORKDIR /app

# CPU wheels: a GPU is not needed to score one image at a time.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . "fastapi>=0.110" "uvicorn>=0.29" "python-multipart>=0.0.9"

# Fetch the ImageNet backbone weights at build time so the container starts offline.
RUN python -c "import torchvision; torchvision.models.wide_resnet50_2(weights='IMAGENET1K_V1')"

RUN useradd --create-home --uid 10001 inspector
USER inspector

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

ENTRYPOINT ["inspector", "serve", "--artifact", "/artifact", "--host", "0.0.0.0", "--port", "8000"]
