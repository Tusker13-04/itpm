from __future__ import annotations

import logging
import os
import shutil
import tempfile

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.schemas import Detection, DetectRequest, DetectResponse
from pipeline.graph import pipeline
from pipeline.state import VisionState

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Open-Vocab Vision Pipeline",
    version="1.0.0",
    description="Grounding DINO + SAM2 + CLIP orchestrated with LangGraph.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    """
    Run the full pipeline on an existing image path.
    """
    state: VisionState = {
        "image_path": request.image_path,
        "prompt": request.prompt,
        "boxes": None,
        "phrases": None,
        "logits": None,
        "masks": None,
        "clip_scores": None,
        "final": None,
        "error": None,
    }

    # Invoke the graph
    result = pipeline.invoke(state)

    if result.get("error") and not result.get("final"):
        raise HTTPException(status_code=422, detail=result["error"])

    detections = [
        Detection(label=d["label"], box=d["box"], score=d["score"])
        for d in (result.get("final") or [])
    ]

    return DetectResponse(
        detections=detections,
        total=len(detections),
        error=result.get("error"),
    )


@app.post("/detect/upload", response_model=DetectResponse)
async def detect_upload(
    file: UploadFile = File(...),
    prompt: str = "person",
):
    """
    Convenience endpoint that accepts an uploaded image file.
    """
    # Persist upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "")[1] or ".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        return await detect(
            DetectRequest(
                image_path=tmp_path,
                prompt=prompt,
            )
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            log.warning("Failed to remove temp file %s", tmp_path)


@app.get("/graph/ascii")
def graph_ascii():
    """
    Return an ASCII representation of the pipeline graph.
    """
    return {"graph": pipeline.get_graph().draw_ascii()}


@app.get("/graph/mermaid")
def graph_mermaid():
    """
    Return Mermaid source for visualizing the pipeline graph.
    """
    return {"mermaid": pipeline.get_graph().draw_mermaid()}


@app.get("/ui", response_class=HTMLResponse)
def ui():
    """
    Minimal web UI: upload image + prompt, run pipeline, view results + graph.
    """
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Open-Vocab Vision Pipeline UI</title>
    <style>
      :root { color-scheme: dark; }
      body { margin: 0; font-family: ui-sans-serif, system-ui, Segoe UI, Roboto, Arial; background:#0b1020; color:#e7e7e7; }
      header { padding: 16px 20px; border-bottom: 1px solid #1f2a44; display:flex; gap:12px; align-items:center; }
      header .badge { font-size:12px; padding:4px 8px; border:1px solid #2b3a61; border-radius:999px; color:#b9c7ff; background:#101a33; }
      main { display:grid; grid-template-columns: 420px 1fr; gap: 16px; padding: 16px; }
      .card { background:#0f1730; border:1px solid #1f2a44; border-radius:14px; padding:14px; }
      .card h2 { margin:0 0 10px; font-size:14px; color:#cdd6ff; letter-spacing:0.2px; }
      label { display:block; font-size:12px; color:#b8c0e0; margin:10px 0 6px; }
      input[type="text"] { width:100%; padding:10px 10px; border-radius:10px; border:1px solid #243153; background:#0b1124; color:#fff; }
      input[type="file"] { width:100%; }
      button { width:100%; margin-top:12px; padding:10px 12px; border-radius:12px; border:1px solid #2a3a66; background:#1b2a55; color:#fff; cursor:pointer; font-weight:600; }
      button:disabled { opacity:0.55; cursor:not-allowed; }
      .row { display:flex; gap:10px; }
      .muted { color:#9aa7d0; font-size:12px; }
      pre { margin:0; padding:12px; background:#0b1124; border:1px solid #223054; border-radius:12px; overflow:auto; max-height: 420px; }
      #graph { background:#0b1124; border:1px solid #223054; border-radius:12px; padding:10px; overflow:auto; }
      .split { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }
      a { color:#9db4ff; text-decoration:none; }
    </style>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
      mermaid.initialize({ startOnLoad: false, theme: "dark" });

      async function loadGraph() {
        const r = await fetch("/graph/mermaid");
        const j = await r.json();
        const code = j.mermaid;
        const el = document.getElementById("graph");
        el.innerHTML = `<pre class="muted" style="margin:0 0 10px">${escapeHtml(code)}</pre><div class="mermaid">${code}</div>`;
        await mermaid.run({ nodes: el.querySelectorAll(".mermaid") });
      }

      function escapeHtml(str) {
        return str.replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
      }

      async function runDetect() {
        const file = document.getElementById("file").files?.[0];
        const prompt = document.getElementById("prompt").value || "person";
        const status = document.getElementById("status");
        const out = document.getElementById("out");
        if (!file) { status.textContent = "Pick an image file first."; return; }

        const btn = document.getElementById("run");
        btn.disabled = true;
        status.textContent = "Running… (first run can take a while while models warm up)";
        out.textContent = "";

        const form = new FormData();
        form.append("file", file);

        const resp = await fetch(`/detect/upload?prompt=${encodeURIComponent(prompt)}`, { method: "POST", body: form });
        const text = await resp.text();
        btn.disabled = false;

        if (!resp.ok) {
          status.textContent = `Error ${resp.status}`;
          out.textContent = text;
          return;
        }
        status.textContent = "Done.";
        try { out.textContent = JSON.stringify(JSON.parse(text), null, 2); }
        catch { out.textContent = text; }
      }

      window.addEventListener("DOMContentLoaded", async () => {
        document.getElementById("run").addEventListener("click", runDetect);
        await loadGraph();
      });
    </script>
  </head>
  <body>
    <header>
      <div style="font-weight:800;">Open‑Vocab Vision Pipeline</div>
      <div class="badge">GroundingDINO + SAM2 + CLIP + LangGraph</div>
      <div style="margin-left:auto" class="muted">
        <a href="/docs" target="_blank">Swagger</a>
      </div>
    </header>
    <main>
      <section class="card">
        <h2>Run detection</h2>
        <label>Image</label>
        <input id="file" type="file" accept="image/*" />
        <label>Prompt (dot-separated)</label>
        <input id="prompt" type="text" value="person . laptop" />
        <button id="run">Run</button>
        <div id="status" class="muted" style="margin-top:10px;"></div>
      </section>
      <section class="split">
        <div class="card">
          <h2>Results</h2>
          <pre id="out" class="muted">Upload an image and click Run.</pre>
        </div>
        <div class="card">
          <h2>Graph</h2>
          <div id="graph" class="muted">Loading graph…</div>
        </div>
      </section>
    </main>
  </body>
</html>
    """


@app.get("/health")
def health():
    """
    Basic health check.
    """
    return {"status": "ok"}


@app.get("/runtime")
def runtime():
    """
    Runtime diagnostics: CUDA availability, device info, CUDA_HOME, versions.
    """
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CUDA_HOME": os.environ.get("CUDA_HOME"),
    }

