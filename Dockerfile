# Используем Python 3.11 внутри контейнера независимо от версии Python
# на хост-машине - это снимает риск несовместимости пакетов faiss-cpu/torch
# с очень новыми версиями Python (см. Project_template.md, раздел "Окружение").
FROM python:3.11-slim

WORKDIR /app

# Системные зависимости, нужные для сборки некоторых Python-пакетов (torch/faiss).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# По умолчанию запускаем консольного бота; для Telegram-бота
# переопределите command в docker-compose.yml или при docker run.
CMD ["python", "scripts/repl_bot.py"]
