FROM python:3.11
WORKDIR /app
COPY requirements.txt ./
# Instalar dependencias de sistema necesarias
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    libssl-dev \
#    libffi-dev \
#    python3-dev \
#    git \
#    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r ./requirements.txt
ENV PYTHONPATH=/app
COPY . /app
RUN chmod +x /app/docker-entrypoint.sh
CMD ["/bin/bash","/app/docker-entrypoint.sh"]
# instalar tzdata para sincronizar hora
RUN apt-get update && apt-get install -y tzdata
ENV TZ=Europe/Madrid
