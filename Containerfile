FROM docker.io/library/python:3

WORKDIR /app

COPY requirements.txt .
COPY embedding.csv .
COPY docsearch.py docsearch.py

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "docsearch.py"]
