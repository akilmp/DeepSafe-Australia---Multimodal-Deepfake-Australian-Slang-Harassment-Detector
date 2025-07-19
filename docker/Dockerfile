FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip && pip install poetry && poetry install --no-dev
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
