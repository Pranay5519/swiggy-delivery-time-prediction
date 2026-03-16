FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-dockers.txt .

RUN pip install --no-cache-dir -r requirements-dockers.txt

COPY app.py .
COPY data_clean_utils.py .
COPY run_information.json .
COPY models/preprocessor.joblib models/preprocessor.joblib

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]