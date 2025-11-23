FROM python:3.10-slim

WORKDIR /app

# 复制整个项目（包含 api_main.py + helper_lib + artifacts）
COPY . /app

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# FastAPI 文档端口
EXPOSE 8002

CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8002"]