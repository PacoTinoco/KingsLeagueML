FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /code

# Copiar archivo de requisitos
COPY ./requirements.txt /code/requirements.txt

# Instalar dependencias necesarias del sistema y Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps && \
    pip install --no-cache-dir -U pip && \
    rm /etc/localtime && \
    ln -s /usr/share/zoneinfo/America/Mexico_City /etc/localtime && \
    pip install --no-cache-dir -r ./requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


COPY ./main.py /code/
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
