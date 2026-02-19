import gradio as gr
import torch
import os

# Try importing unsloth, fallback to standard transformers for Mac/CPU
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

# Configuration
ADAPTER_PATH = "lora_model_final" # We are pushing the model directly to the Space
BASE_MODEL_ID = "google/gemma-2b-it" # Standard base model

def load_model():
    print("Loading model...")
    if UNSLOTH_AVAILABLE and torch.cuda.is_available():
        print("🚀 Using Unsloth on GPU")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = ADAPTER_PATH if os.path.exists(ADAPTER_PATH) else BASE_MODEL_ID,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer, "cuda"
    
    else:
        # Fallback for Mac (MPS) or CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🍎 Using Standard Transformers on {device.upper()}")
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=os.getenv("HF_TOKEN"))
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            device_map="auto" if device != "cpu" else None,
            token=os.getenv("HF_TOKEN")
        )
        
        # Load Adapter if available (Local folder or HF Repo)
        try:
            if os.path.exists(ADAPTER_PATH):
                print(f"Loading Adapter from local folder: {ADAPTER_PATH}")
                # We skip the auto-generating config hack as it should be correct from training now
                model = PeftModel.from_pretrained(model, ADAPTER_PATH)
                print("✅ Adapter loaded successfully.")
            else:
                print(f"Local adapter not found. Trying to load '{ADAPTER_PATH}' from Hugging Face Hub...")
                model = PeftModel.from_pretrained(model, ADAPTER_PATH)
                print(f"Successfully loaded adapter from Hub: {ADAPTER_PATH}")
        except Exception as e:
            print(f"Warning: Could not load adapter: {e}")
            print("Running with Base Model only.")
            
        if device == "mps":
            model.to("mps")
            
        return model, tokenizer, device

# Load model globally
model, tokenizer, device = load_model()

# STRICT PROMPT from train_final.ipynb
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# --- UI Redesign ---
# --- UI Redesign ---
# --- UI Redesign ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@300;400;600;700&display=swap');

body {
    background-image: url('https://images.unsplash.com/photo-1500937386664-56d1dfef3854?q=80&w=2938&auto=format&fit=crop'); /* Wheat field / Farm landscape */
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Work Sans', sans-serif;
    color: #2d3436;
}

.gradio-container {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(10px);
    max-width: 1000px !important;
    margin: 40px auto !important;
    border-radius: 24px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.5);
    padding-top: 0 !important;
    overflow: hidden;
}

#main-card {
    background: transparent;
    border: none;
    box-shadow: none;
    padding: 0 !important;
}

#header-image {
    height: 300px;
    background-image: url('https://images.unsplash.com/photo-1625246333195-78d9c38ad449?q=80&w=2670&auto=format&fit=crop'); /* Farmer/Garden close up */
    background-size: cover;
    background-position: center;
    position: relative;
    border-radius: 24px 24px 0 0;
}

#header-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 100%;
    background: linear-gradient(to bottom, rgba(0,0,0,0.1), rgba(0,0,0,0.7));
    display: flex;
    align-items: flex-end;
    padding: 40px;
}

.title-container {
    color: white;
}

.title-text {
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 8px;
    letter-spacing: -1px;
    text-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.subtitle-text {
    font-size: 18px;
    opacity: 0.95;
    font-weight: 400;
    color: #e8f5e9; /* Light green tint */
}

.tag-badge {
    background: #4caf50; /* Fresh Green */
    color: white;
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    display: inline-block;
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
}

#content-area {
    padding: 48px;
    background: white;
}

.input-label {
    font-size: 14px;
    font-weight: 700;
    color: #2e7d32; /* Dark Green */
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
    display: block;
}

#input-box textarea {
    background: #f1f8e9 !important; /* Very light green bg */
    border: 2px solid #c5e1a5 !important;
    border-radius: 16px !important;
    color: #33691e !important;
    padding: 20px !important;
    font-size: 18px !important;
    transition: all 0.3s ease;
}

#input-box textarea:focus {
    border-color: #558b2f !important;
    background: white !important;
    box-shadow: 0 0 0 4px rgba(85, 139, 47, 0.1);
}

#action-btn {
    background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    padding: 20px 32px !important;
    margin-top: 16px !important;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 8px 20px rgba(76, 175, 80, 0.25);
}

#action-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(76, 175, 80, 0.35);
}

.output-container {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 20px;
    padding: 32px;
    height: 100%;
    box-shadow: 0 4px 20px rgba(0,0,0,0.03);
}

.output-label {
    font-size: 13px;
    font-weight: 800;
    color: #f57f17 !important; /* Harvest gold/orange */
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 20px;
    display: block;
}

.generated-text {
    font-size: 18px;
    line-height: 1.8;
    color: #2d3436 !important;
}

.generated-text * {
    color: #2d3436 !important;
}

.plant-card {
    background: #f1f8e9;
    border-radius: 16px;
    padding: 20px;
    margin-top: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    border: 1px solid #c5e1a5;
}

.plant-icon {
    width: 56px;
    height: 56px;
    background: white;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
"""

def format_output(plant_name, response):
    return f"""
    <div>
        <span class="output-label">Detailed Analysis</span>
        <div class="generated-text">{response}</div>
        
        <div class="plant-card">
            <div class="plant-icon">🌿</div>
            <div>
                <div style="font-size: 12px; color: #558b2f; text-transform: uppercase; letter-spacing: 1px; font-weight: 700;">Subject</div>
                <div style="font-size: 20px; font-weight: 700; color: #2d3436;">{plant_name.title() if plant_name else "Unknown"}</div>
            </div>
            <div style="margin-left: auto; text-align: right;">
                <div style="font-size: 12px; color: #558b2f; text-transform: uppercase; letter-spacing: 1px; font-weight: 700;">Status</div>
                <div style="font-size: 15px; font-weight: 700; color: #43a047;">Active Reference</div>
            </div>
        </div>
    </div>
    """

def ui_predict(plant_name, progress=gr.Progress()):
    print(f"DEBUG: Received input: {plant_name}")
    if not plant_name:
        return format_output("", "Please enter a plant name to get started.")
    try:
        progress(0.1, desc="Preparing model inputs...")
        # Mocking the generation for UI testing if model isn't loaded, or using real generation
        if 'model' in globals():
             print("DEBUG: Model is loaded. Tokenizing...")
             
             # STRICT INSTRUCTION matching training
             instruction = f"List strictly the best companion plants for {plant_name}. Do not add any others."
             
             inputs = tokenizer(
                [
                    alpaca_prompt.format(
                        instruction,
                        "",
                        "",
                    )
                ], return_tensors = "pt").to(device)
             print(f"DEBUG: Input tokens shape: {inputs.input_ids.shape}")

             progress(0.3, desc="Generating response (this may take a moment)...")
             print("DEBUG: generating...")
             
             # Reduced max_new_tokens and added repetition penalties
             outputs = model.generate(
                **inputs, 
                max_new_tokens = 256, 
                use_cache = True,
                repetition_penalty = 1.2,
                no_repeat_ngram_size = 3,
                temperature = 0.5, # Lower temperature for stricter factual recall
                top_p = 0.9,
                do_sample = True
             )
             print("DEBUG: Generation complete. Decoding...")
             
             raw_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
             # Also strip explicit <eos> if it leaked through
             raw_response = raw_response.replace("<eos>", "").strip()
             
             print(f"DEBUG: Raw response length: {len(raw_response)}")
        
             # Extract response after "### Response:"
             if "### Response:" in raw_response:
                 response_text = raw_response.split("### Response:")[1].strip()
             else:
                 response_text = raw_response
            
             progress(1.0, desc="Done!")
             return format_output(plant_name, response_text)
        else:
             print("ERROR: Model not found in globals!")
             return format_output(plant_name, "Error: Model not loaded.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return format_output(plant_name, f"Error generating response: {str(e)}")


with gr.Blocks(css=custom_css, title="Companion Planting AI") as app:
    with gr.Column(elem_id="main-card"):
        # Header Image Area
        gr.HTML("""
        <div id="header-image">
            <div id="header-overlay">
                <div class="title-container">
                    <span class="tag-badge">AI Gardening Assistant</span>
                    <div class="title-text">Companion Planting Guide</div>
                    <div class="subtitle-text">Expert advice for your garden powered by Gemma</div>
                </div>
            </div>
        </div>
        """)
        
        # Main Content Area
        with gr.Row(elem_id="content-area"):
            with gr.Column(scale=4):
                gr.HTML('<span class="input-label">What are you growing?</span>')
                input_text = gr.Textbox(
                    placeholder="e.g., Temperature-sensitive Tomato, Basil...", 
                    elem_id="input-box",
                    show_label=False,
                    lines=1
                )
                submit_btn = gr.Button("Find Companions", elem_id="action-btn")
            
            with gr.Column(scale=6):
                output_html = gr.HTML(
                    value=format_output("", "Enter a plant name to see its best companions and gardening tips here."),
                    elem_classes="output-container"
                )

    submit_btn.click(fn=ui_predict, inputs=input_text, outputs=output_html)
    input_text.submit(fn=ui_predict, inputs=input_text, outputs=output_html)

if __name__ == "__main__":
    # share=True is not supported on specific environments (like HF Spaces), so we default to False unless specified.
    # Users can enable it manually if running locally and they want a public link.
    app.launch()
