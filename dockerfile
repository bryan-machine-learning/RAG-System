FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Ollama runs on the host machine, not inside the container
ENV LLM_BASE_URL=http://host.docker.internal:11434/v1

EXPOSE 8000
CMD ["uvicorn", "--app-dir", "src", "api:app", "--host", "0.0.0.0", "--port", "8000"]