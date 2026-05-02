import torch
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from core.logger_ws import ws_logger

# ==========================================
# PROMPTS DE LOS AGENTES (Definidos por el usuario)
# ==========================================

PROMPT_AGENTE_1 = """You are an elite Viral Content Strategist for long-form Shorts (exactly 2 MINUTES long). Your only goal is viewer retention.
You will receive a raw topic or idea. Your task is to output ONLY two things:
1. THE 3-SECOND HOOK: A visual and auditory concept for the very first scene that creates instant, overwhelming curiosity.
2. THE 2-MINUTE PREMISE: A detailed 4-sentence summary of the story. It MUST include an escalating conflict, a major PLOT TWIST at the 1-minute mark, a climax, and a satisfying payoff. A simple setup is not enough; the story must justify 120 seconds of attention."""

PROMPT_AGENTE_2 = """You are a Master Screenwriter. You will receive a "Hook and Premise". 
Your task is to write a highly engaging script specifically timed for a 2-MINUTE video.
RULES:
1. TARGET LENGTH: Generate exactly 20 to 24 distinct SCENES.
2. THE 6-SECOND RULE (CRITICAL): EVERY single scene represents exactly 6 SECONDS of video. To prevent awkward silences, you MUST write enough dialogue or voiceover to fill 4 to 6 seconds per scene (approximately 12 to 18 words per scene).
3. DIALOGUE CONTINUITY: Do NOT write disconnected one-liners. Dialogue must flow logically from one scene to the next. If Scene 1 asks a question or starts a thought, Scene 2 must answer or continue it. The narrative tissue must be seamless.
4. IMMEDIATE HOOK: Scene 1 must begin with the protagonist speaking immediately. No slow visual intros.
5. CONTEXT & PACING: At least 30% of scenes must be "ACTION/ESTABLISHING SCENES" (using Voiceover instead of on-screen talking) to show the environment.
6. FORMAT: Number each scene. Clearly label if a scene is [ON-SCREEN DIALOGUE] or [ACTION/VOICEOVER]."""

PROMPT_AGENTE_3 = """You are a strict Continuity Director. You will receive a 2-minute Script. Your job is to extract the "Production Bible" (Characters and Locations) and annotate the script.
CRITICAL RULES FOR CHARACTERS:
1. Protagonist(s): Give them highly specific, distinct physical traits (Age, ethnicity, exact clothing item and color, hairstyle).
2. NPCs (Secondary Characters): ANY character who is not the main focus MUST be visually muted (generic, dull clothing like "faded grey shirt", no distinguishing features). 
3. NPC Action: NPCs exist ONLY to push the protagonist's story forward.

OUTPUT FORMAT: Return a structured text containing the Production Bible first, followed by the Annotated Script where character names are completely replaced by their exact physical descriptions."""

PROMPT_AGENTE_4 = """You are an elite AI Video Prompt Engineer. You will receive an "Annotated Script" and a "Production Bible". Convert EVERY SCENE into a strict JSON array of video generation prompts. Do not summarize.

CRITICAL DIRECTING RULES (THE 6-SECOND WINDOW):
Every prompt you generate will create exactly a 6-SECOND video clip. You must describe continuous cinematic motion, micro-expressions, or camera movements that fill this entire 6-second window so the video does not feel static.

TYPE A: ON-SCREEN DIALOGUE (The Anti-Ventriloquist Protocol)
- Use ONLY "Cinematic Close-Up" or "Medium Shot" focused entirely on the speaking character.
- The speaking character must explicitly be described as "moving their lips and speaking continuously".
- Passive characters in frame MUST be described as "completely frozen, mouth closed, in silence."

TYPE B: ACTION / ESTABLISHING SCENES
- Use "Wide Shot", "Tracking Shot", etc. Show the environment and physical actions.
- EVERY character in these scenes must have their MOUTH CLOSED. 

REQUIRED PROMPT STRUCTURE:
"[Camera angle]. [Continuous 6-second action/camera movement]. [Exact Location]. [Exact Character Physical Description]. [Mouth status: 'moving their lips and speaking continuously' OR 'mouth closed, not speaking']. They say in Spanish: '[dialogue]'. Voice of [NAME] ([EXACT GENDER] voice): Studio-quality, crystal clear articulation, no background noise, no distortion, [Tone, volume, rhythm]."

OUTPUT FORMAT:
Return ONLY a valid JSON array of strings. No markdown formatting, no explanations."""


# ==========================================
# LÓGICA DEL PIPELINE Y MODELO
# ==========================================

def load_llm():
    ws_logger.log("🧠 Cargando LLM (Agentes) en VRAM (4-bits)...")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto"
    )
    return model, tokenizer


def run_inference(model, tokenizer, prompt_system, prompt_user):
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]

    # Aplicar template y tokenizar de forma segura para evitar KeyError: 'shape'
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    pad_token = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, int) else 128001

    outputs = model.generate(
        **inputs,
        max_new_tokens=2500,  # Aumentado un poco porque 24 escenas pueden ser largas
        eos_token_id=terminators,
        pad_token_id=pad_token
    )

    input_length = inputs["input_ids"].shape[-1]
    # clean_up_tokenization_spaces=False evita el warning molesto en la consola
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)

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

        # Extractor de JSON a prueba de balas (ignora saludos o markdown extra de Llama)
        start_idx = json_raw.find('[')
        end_idx = json_raw.rfind(']')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_clean = json_raw[start_idx:end_idx + 1]
            try:
                prompts_array = json.loads(json_clean)
                ws_logger.log(f"✅ JSON validado: {len(prompts_array)} escenas de 6 segundos generadas.")
                return prompts_array
            except json.JSONDecodeError as e:
                ws_logger.log(f"❌ Error al decodificar el JSON: {str(e)}")
                print(f"\n--- JSON CRUDO ---\n{json_raw}\n------------------\n")
                raise Exception("El formato interno del JSON generado por la IA es inválido.")
        else:
            print(f"\n--- RESPUESTA CRUDA ---\n{json_raw}\n------------------\n")
            raise Exception("La IA no devolvió un arreglo JSON ([...]).")

    finally:
        # CRÍTICO: Liberar VRAM para que CogVideoX pueda entrar
        ws_logger.log("🧹 Limpiando VRAM: Descargando LLM...")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()