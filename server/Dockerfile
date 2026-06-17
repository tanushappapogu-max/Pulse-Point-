FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download LocateAnything-3B weights so cold starts are fast.
# Falls back gracefully if HF_TOKEN is not set or no GPU available.
RUN python -c "
from locate_model import _load
try:
    _load()
    print('LocateAnything-3B pre-loaded')
except Exception as e:
    print(f'LocateAnything pre-load skipped: {e}')
"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
