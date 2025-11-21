FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY models.py .
COPY database.py .
COPY redis_client.py .
COPY llm_client.py .
COPY mcp_client.py .
COPY qdrant_service.py .
COPY content_processor.py .
COPY markdown_to_tiptap.py .
COPY constants/ ./constants/

EXPOSE 8080

CMD ["python", "main.py"]
