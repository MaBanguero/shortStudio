import os
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import traceback
import asyncio
import concurrent.futures

from core.logger_ws import ws_logger
from core.agent_pipeline import generate_script_json
from core.video_engine import generate_video_clips

app = FastAPI()

# Asegurar que el directorio de outputs existe
os.makedirs("outputs", exist_ok=True)


class VideoRequest(BaseModel):
    topic: str


@app.get("/")
async def get():
    with open("frontend/index.html", "r") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_logger.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Mantener la conexión viva
    except WebSocketDisconnect:
        ws_logger.disconnect(websocket)


def run_full_pipeline(topic: str, project_id: str):
    """Esta función corre de forma síncrona en un ThreadPool"""
    try:
        # 1. Pipeline de Texto
        prompts_array = generate_script_json(topic)

        # 2. Pipeline de Video
        generate_video_clips(prompts_array, project_id)

        ws_logger.log(f"🎉 ¡Proyecto {project_id} finalizado exitosamente!")
    except Exception as e:
        error_trace = traceback.format_exc()
        ws_logger.log(f"❌ ERROR CRÍTICO: {str(e)}")
        print(f"\n--- DETALLE DEL ERROR ---\n{error_trace}\n-----------------------\n")


@app.post("/generate")
async def generate_video(request: VideoRequest):
    project_id = str(uuid.uuid4())[:8]

    # Ejecutamos el pipeline pesado en un hilo separado para no bloquear FastAPI
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, run_full_pipeline, request.topic, project_id)

    return {"status": "started", "project_id": project_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)