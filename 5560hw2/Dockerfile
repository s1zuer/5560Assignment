# 使用轻量 Python 环境
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt && pip install python-multipart

# 暴露端口
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]