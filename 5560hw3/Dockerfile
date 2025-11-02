# 基础镜像：Python 3.10 精简版（和上次保持一致）
FROM python:3.10-slim

# 在容器里创建工作目录
WORKDIR /app

COPY pyproject.toml /app/pyproject.toml


RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .

COPY . /app


EXPOSE 8002

CMD ["uvicorn", "api_main_gan:app", "--host", "0.0.0.0", "--port", "8002"]