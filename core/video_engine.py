import torch
import gc
from diffusers import CogVideoXPipeline
from transformers import BitsAndBytesConfig
from core.logger_ws import ws_logger


def load_video_model():
    ws_logger.log("🎥 Cargando CogVideoX-5b en VRAM (4-bits)...")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        quantization_config=quant_config,
        device_map="auto"
    )
    return pipe


def generate_video_clips(prompts_array: list, project_id: str):
    pipe = load_video_model()

    try:
        total_clips = len(prompts_array)
        for index, prompt in enumerate(prompts_array):
            scene_num = index + 1
            ws_logger.log(f"🎞️ Generando Clip {scene_num}/{total_clips} (6 segundos)...")
            ws_logger.log(f"📝 Prompt: {prompt[:100]}...")  # Imprime un extracto

            video = pipe(
                prompt=prompt,
                num_frames=49,  # 49 frames en CogVideoX equivalen a ~6 seg a 8fps
                num_inference_steps=25,
                guidance_scale=6.0
            ).frames[0]

            # Guardar clip
            # Nota: Necesitas la utilidad export_to_video de diffusers.utils
            from diffusers.utils import export_to_video
            output_path = f"outputs/{project_id}_scene_{scene_num:02d}.mp4"
            export_to_video(video, output_path, fps=8)
            ws_logger.log(f"💾 Clip {scene_num} guardado en {output_path}")

    finally:
        # CRÍTICO: Liberar VRAM
        ws_logger.log("🧹 Limpiando VRAM: Descargando CogVideoX...")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()