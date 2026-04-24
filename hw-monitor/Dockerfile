FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hw_monitor.py .

CMD ["python", "hw_monitor.py"]
