FROM python:3.10-slim

WORKDIR /app
COPY app.py .

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install diffusers transformers accelerate gradio

EXPOSE 7860

CMD ["python", "app.py"]
