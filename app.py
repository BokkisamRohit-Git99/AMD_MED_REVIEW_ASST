import os
import json
import torch
import requests
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# 🌟 HUGGING FACE ZEROGPU COMPATIBILITY
# This allows your app to use Hugging Face's free A100/H200 GPU cluster on-demand
try:
    import spaces
except ImportError:
    # Fallback simulation if you test this file locally on your CPU
    class spaces:
        @staticmethod
        def GPU(func):
            return func

# Configuration (Replace with your actual repo target)
MODEL_ID = "Bokkisam-Rohit24/Qwen2.5-3B-Medical-PV-Extract"

print("⏳ Initializing system infrastructure...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# For ZeroGPU, load the model globally onto the CPU. 
# The @spaces.GPU decorator will automatically teleport it to a GPU when called!
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    device_map="cpu" 
)
print("🟢 System Ready! Launching Gradio Portal...")

# ⚡ THIS DECORATOR IS THE SECRET SAUCE: Allocates a physical GPU container instantly
@spaces.GPU
def process_extraction(narrative, hf_token):
    # 1. Enforce Token Presence
    if not hf_token.strip():
        return {"Error": "Security Barrier: Please provide your personal Hugging Face Token in the input block."}
    
    # 2. Authenticate token dynamically against Hugging Face identity gates
    headers = {"Authorization": f"Bearer {hf_token.strip()}"}
    hf_verify = requests.get("https://huggingface.co/api/whoami-v2", headers=headers, timeout=5)
    
    if hf_verify.status_code != 200:
        return {"Error": "Authentication Failed: The provided token was rejected by Hugging Face."}
    
    # 3. Target safe execution device (dynamically scales to 'cuda' via ZeroGPU context)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 4. Format prompt structure
    messages = [
        {"role": "system", "content": "Extract PV data into a single-line flattened JSON string."},
        {"role": "user", "content": narrative}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 5. Fast GPU Tensor Execution
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=False,  # Factual, greedy decoding
            eos_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 6. Attempt structural parse output conversion
    try:
        return json.loads(generated_text.strip())
    except Exception:
        return {"Raw Model String Output": generated_text.strip()}


# --- GRADIO UI LAYOUT DESIGN ---
with gr.Blocks(theme=gr.themes.Soft(), title="PV Intelligence Hub") as demo:
    gr.Markdown("# 🩺 Enterprise Pharmacovigilance Extraction Hub")
    gr.Markdown("Deploys multi-threaded clinical extraction pipelines natively through localized UI blocks.")
    gr.HTML("<hr style='border: 1px solid #e5e7eb;' />")
    
    with gr.Row():
        # Left Panel: User Configurations & Inputs
        with gr.Column(scale=1):
            gr.Markdown("### 🔑 Authentication")
            token_input = gr.Textbox(
                label="Personal Hugging Face Token", 
                placeholder="Paste hf_... here", 
                type="password",
                info="Your token authenticates your session allocation footprint and is processed safely in-memory."
            )
            
            gr.Markdown("### 📋 Clinical Narrative Source")
            narrative_input = gr.Textbox(
                label="Patient Case Record Text", 
                placeholder="Paste raw narrative details here...", 
                lines=10
            )
            
            submit_btn = gr.Button("Extract Structured Profiles", variant="primary")
            
        # Right Panel: Results View
        with gr.Column(scale=1):
            gr.Markdown("### ⚡ Structured JSON Output Profile")
            json_output = gr.JSON(label="Parsed Matrix Variables Result View")

    # Gradio Native Examples Matrix (Replaces the drop-down template selection logic)
    gr.Markdown("### 💡 Quick-Load Mock Clinical Scenarios")
    gr.Examples(
        examples=[
            [
                "A 62-year-old female patient with a history of chronic plaque psoriasis was started on Secukinumab (Cosentyx) 300mg weekly. Following the fourth loading dose, she experienced severe watery diarrhea and abdominal cramping.",
                "" # Leaves token blank initially for the user to paste their own
            ]
        ],
        inputs=[narrative_input, token_input]
    )

    # Core Event Wireframe Routing
    submit_btn.click(
        fn=process_extraction,
        inputs=[narrative_input, token_input],
        outputs=[json_output]
    )

# Required execution check layer for local versus Space environments
if __name__ == "__main__":
    demo.launch()