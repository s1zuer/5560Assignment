# 使用 Python 3.10 的精简版镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 拷贝依赖声明文件（pyproject.toml）
COPY pyproject.toml /app/pyproject.toml

# 安装构建系统工具并安装项目依赖
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .

# 拷贝所有源代码
COPY . /app

# 暴露 FastAPI 端口
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]