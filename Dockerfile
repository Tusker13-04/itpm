FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir langgraph-cli[inmem]

COPY . .

EXPOSE 8123

# Run LangGraph Studio backend by default
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8123"]
