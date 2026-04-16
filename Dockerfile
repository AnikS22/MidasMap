# MidasMap Gradio app — CPU. For GPU, use a CUDA base and install torch+cu* instead.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY huggingface-space/requirements-space.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY src ./src
COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh

ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860

ENTRYPOINT ["/docker_entrypoint.sh"]
CMD []
