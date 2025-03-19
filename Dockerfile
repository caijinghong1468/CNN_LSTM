# 選擇適合的 Python 基礎映像，例如 Python 3.9
FROM python:3.12.0

# 設定工作目錄
WORKDIR /app

# 複製當前目錄的內容到 Docker 容器內
COPY . /app

# 安裝 Python 依賴（如果你有 requirements.txt）
RUN pip install --no-cache-dir -r requirements.txt

# 指定 Python 可執行檔
CMD ["python", "vgg19/main.py"]