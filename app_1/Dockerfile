# Используйте официальный образ Python
FROM python:3.13-slim

# Установите системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app /app
WORKDIR /app
EXPOSE 8501
CMD ["streamlit", "run", "main.py"]