FROM python:3.11-slim

WORKDIR /ui

RUN pip install --no-cache-dir \
        streamlit==1.36.0 \
        requests==2.32.3 \
        pandas==2.2.2

COPY ui/streamlit_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
