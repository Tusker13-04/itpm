# Open-Vocab Vision Pipeline using LangGraph

This repository contains a vision pipeline that combines **Grounding DINO** (open-vocabulary object detection), **SAM2** (segmentation), and **CLIP** (re-scoring) orchestrated via **LangGraph**.

## Setup & Installation

1. Create and activate a Python virtual environment.
2. Install your local PyTorch wheel first (e.g., CUDA 12.6 build):
   ```bash
   pip install path/to/torch-wheel.whl torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the model weights:
   ```bash
   python download_weights.py
   ```

## Running the Pipeline via LangGraph Studio

This pipeline is designed to be run directly through the LangGraph UI/Studio (no separate FastAPI or Uvicorn server is needed).

1. Ensure your `.env` file contains your LangSmith API key:
   ```env
   LANGSMITH_API_KEY=your_key_here
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=itpm-vision-pipeline
   PYTHONPATH=.
   ```
2. Start the LangGraph development server:
   ```bash
   langgraph dev
   ```
3. In the LangGraph Studio UI:
   - Click the **`+`** icon in the chat box to upload an image.
   - Type your detection prompt in the text box (e.g., `person . car . laptop`).
   - Submit the message. The pipeline will process the image and reply with the detections and confidence scores.
