FROM python:3.10-slim

WORKDIR /app

# 复制依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 8000

# 启动 FastAPI
CMD ["uvicorn", "rl_llm.api_server:app", "--host", "0.0.0.0", "--port", "8000"]