# 帯付けくん — Hugging Face Spaces (Docker SDK) 用
FROM python:3.13-slim

# PyMuPDF/Pillowの動作に必要な最小限のライブラリ
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spacesはroot以外（UID 1000）での実行を推奨
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# 進捗はメモリ上で管理するため worker は必ず1つ。並行リクエストは threads で捌く。
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "120", "--bind", "0.0.0.0:7860", "app:app"]
