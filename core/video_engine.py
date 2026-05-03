import torch
import gc
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from core.logger_ws import ws_logger


def load_video_model():
    ws_logger.log("🎥 Cargando CogVideoX-5b (Precisión Nativa bfloat16 + Optimizaciones de VRAM)...")

    # Cargamos el modelo SIN cuantización en el formato más eficiente
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        torch_dtype=torch.bfloat16
        # Nota: No usamos device_map="auto" aquí porque interfiere con el cpu_offload
    )

    # --- OPTIMIZACIONES CRÍTICAS PARA SOBREVIVIR EN 24GB VRAM ---
    ws_logger.log("⚙️ Activando CPU Offloading y VAE Tiling...")
    pipe.enable_model_cpu_offload()  # Mueve bloques inactivos a la RAM
    pipe.vae.enable_slicing()  # Evita OOM al decodificar
    pipe.vae.enable_tiling()  # Procesa las imágenes por mosaicos

    return pipe


def generate_video_clips(prompts_array: list, project_id: str):
    pipe = load_video_model()

    try:
        total_clips = len(prompts_array)
        for index, scene in enumerate(prompts_array):
            scene_num = index + 1
            ws_logger.log(f"🎞️ Generando Clip {scene_num}/{total_clips} (6 segundos)...")

            # --- MANEJO INTELIGENTE DEL TIPO DE DATO ---
            if isinstance(scene, dict):
                # Si la IA devolvió un diccionario con llaves
                prompt_text = f"{scene.get('camera', '')}. {scene.get('action', '')}. Location: {scene.get('location', '')}. Character: {scene.get('character', '')}. {scene.get('mouth', '')}. {scene.get('dialogue', '')}."
            else:
                # Si la IA devolvió un string de texto directo (como lo ordenaba el prompt)
                prompt_text = str(scene).strip()
            # ------------------------------------------

            ws_logger.log(f"📝 Prompt: {prompt_text[:80]}...")

            # Generación del video
            video = pipe(
                prompt=prompt_text,
                num_frames=49,
                num_inference_steps=25,
                guidance_scale=6.0
            ).frames[0]

            output_path = f"outputs/{project_id}_scene_{scene_num:02d}.mp4"
            export_to_video(video, output_path, fps=8)
            ws_logger.log(f"💾 Clip {scene_num} guardado con éxito.")

    finally:
        ws_logger.log("🧹 Limpiando VRAM: Descargando CogVideoX...")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()