FROM vllm-gaudi:1.24
COPY habana.gpg /usr/share/keyrings/habana.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/habana.gpg] https://vault.habana.ai/artifactory/debian noble main" > /etc/apt/sources.list.d/habanalabs.list && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y habanalabs-qual && rm -rf /var/lib/apt/lists/*
