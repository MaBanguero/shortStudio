import torch
import json
import re
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from core.logger_ws import ws_logger

# Tus prompts originales
PROMPT_AGENTE_1 = """You are an elite Viral Content... [Inserta tu prompt exacto aquí]"""
PROMPT_AGENTE_2 = """You are a Master Screenwriter... [Inserta tu prompt exacto aquí]"""
PROMPT_AGENTE_3 = """You are a strict Continuity Director... [Inserta tu prompt exacto aquí]"""
PROMPT_AGENTE_4 = """You are an elite AI Video Prompt Engineer... [Inserta tu prompt exacto aquí]"""


def load_llm():
    ws_logger.log("🧠 Cargando LLM (Agentes) en VRAM (4-bits)...")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Asegúrate de tener acceso en HuggingFace o usa uno abierto

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")
    return model, tokenizer


def run_inference(model, tokenizer, prompt_system, prompt_user):
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
        model.device)
    outputs = model.generate(input_ids, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response


def generate_script_json(topic: str):
    model, tokenizer = load_llm()

    try:
        ws_logger.log(f"🕵️‍♂️ Agente 1 (Estratega): Generando Hook y Premisa para '{topic}'...")
        hook_premise = run_inference(model, tokenizer, PROMPT_AGENTE_1, f"Topic: {topic}")

        ws_logger.log("✍️ Agente 2 (Guionista): Escribiendo guion de 20-24 escenas...")
        script = run_inference(model, tokenizer, PROMPT_AGENTE_2, hook_premise)

        ws_logger.log("🎬 Agente 3 (Director): Extrayendo Biblia de Producción y anotando guion...")
        annotated_script = run_inference(model, tokenizer, PROMPT_AGENTE_3, script)

        ws_logger.log("🤖 Agente 4 (Prompt Engineer): Generando JSON estricto para CogVideoX...")
        json_raw = run_inference(model, tokenizer, PROMPT_AGENTE_4, annotated_script)

        # Limpiar posible markdown en la salida del JSON
        json_clean = re.sub(r'```json\n|\n```|```', '', json_raw).strip()
        prompts_array = json.loads(json_clean)
        ws_logger.log(f"✅ JSON validado: {len(prompts_array)} escenas generadas.")

        return prompts_array

    finally:
        # CRÍTICO: Liberar VRAM
        ws_logger.log("🧹 Limpiando VRAM: Descargando LLM...")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()