# Pinned to the Python the running image has (INF-07). 3.12 while the api and
# the gateway are on 3.11: it only needs pynvml, redis and SQLAlchemy, and
# moving it is a rebuild with nothing to gain, so it stays where it runs.
FROM python:3.12.16-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hw_monitor.py .

CMD ["python", "hw_monitor.py"]
