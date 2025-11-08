FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY models.py .
COPY database.py .
COPY redis_client.py .

EXPOSE 8080

CMD ["python", "main.py"]
