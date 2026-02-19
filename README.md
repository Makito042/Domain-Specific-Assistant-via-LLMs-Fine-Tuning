---
title: Companion Planting Assistant
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.19.2
python_version: "3.10"
app_file: app.py
pinned: false
license: other
models:
  - google/gemma-2b-it
dataset:
  - companion_plants.csv
tags:
  - companion-planting
  - agriculture
  - gemma
  - unsloth
  - lora
---

# Companion Plants Assistant 🌿

A domain-specific AI assistant fine-tuned to provide expert advice on companion planting. This model helps gardeners specifically identify which plants grow well together to improve yields and repel pests.

## 🚀 Project Overview

- **Domain**: Agriculture / Gardening (Companion Planting)
- **Model**: Gemma-2b-it (Fine-tuned using QLoRA)
- **Dataset**: Custom dataset derived from `companion_plants.csv`
- **Technique**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA

## 📂 Repository Structure

- `train.ipynb`: The complete end-to-end notebook for Google Colab. Covers data preprocessing, training, evaluation, and interactive inference.
- `app.py`: A standalone Gradio application for running the inference interface..
- `companion_plants.csv`: The source dataset.

## 🛠️ How to Run on Google Colab

1. Open `train.ipynb` in Google Colab.
2. Enable GPU: Runtime > Change runtime type > T4 GPU.
3. Upload `companion_plants.csv`  
   - If you upload `companion_plants.csv`, the notebook will process it for you.
4. Run all cells.
5. The final cell will launch a public Gradio link where you can chat with the assistant.

## 📊 Dataset & Preprocessing

The dataset is transformed from a structured CSV into Instruction-Response pairs:
- **Instruction**: "What plants does Tomato help grow better?"
- **Response**: "Tomato is a beneficial companion for: Asparagus, Basil, Beans..."

We aggregate "helps" and "helped_by" relationships to create comprehensive answers.

## ⚙️ Fine-Tuning Details

- **Library**: `unsloth` (Optimized for speed and memory)
- **Base Model**: `unsloth/gemma-2b-it-bnb-4bit`
- **LoRA Config**: Rank=16, Alpha=16
- **Hyperparameters**:
    - Batch Size: 2
    - Gradient Accumulation: 4
    - Learning Rate: 2e-4
    - **Scheduling**: Linear
    - **Epochs**: 15 (Aggressive training for factual recall)
- **Strategy**:
    - **Strict Instruction Tuning**: The model is trained with instructions like *"List strictly the plants..."* to enforce concise, list-based output without hallucination.
    - **EOS Tokens**: Explicit `<eos>` tokens are used to teach the model exactly when to stop generating.
    - **Bi-directional Augmentation**: The dataset is processed to learn both "A helps B" and "B is helped by A" relationships.

## 📈 Performance & Evaluation

The final model is evaluated using both qualitative checks and quantitative metrics:

### Quantitative Metrics (on `train_final.ipynb`):
- **BLEU**: Measures n-gram precision (High scores indicate exact matching of plant lists).
- **ROUGE-L**: Measures longest common subsequence (Important for list recall).
- **Perplexity**: Measures how well the model predicts the next token.

### Qualitative Analysis:
- **Baseline**: The base model often gives generic gardening advice or hallucinates plants not in the database.
- **Fine-Tuned**: The fine-tuned model provides specific, accurate lists of companion plants derived directly from the dataset.

#### Example Conversation:
**User**: "What are the best companion plants for Tomatoes?"
**Assistant**: "Best companions for Tomatoes: Basil, Chives, Marigolds, Parsley." (Exact match to ground truth)

## 🖥️ User Interface

The project includes a Gradio web interface that accepts a plant name and returns its best companions.

## 💻 Running Locally (Mac/Windows/Linux)

1.  **Download your model**:
    - Download `lora_model.zip` from Colab and unzip it into the project folder. You should see a folder named `lora_model`.

2.  **Install dependencies**:
    ```bash
    pip3 install -r requirements.txt
    ```

3.  **Run the App**:
    - **Option A (Interactive Login)**: Run `huggingface-cli login` first.
    - **Option B (Direct Token)**: Run with your token directly:
      ```bash
      HF_TOKEN=hf_your_token_here python3 app.py
      ```
    - On Mac, it will automatically use Metal (MPS) acceleration.
    - On Windows with NVIDIA, it will use CUDA.

## ☁️ Deploying to Hugging Face Spaces
1.  **Push Model to Hub**:
    - In your Colab notebook, run:
      ```python
      model.push_to_hub("your-username/your-model-name", token="...")
      ```
2.  **Create a Space**:
    - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
    - Choose **Gradio** as the SDK.
3.  **Upload Files**:
    - Upload `app.py` and `requirements.txt` to your Space.
4.  **Configure**:
    - In your Space settings, add a simplified **Variable**:
      - `Adapter_REPO`: `your-username/your-model-name`
    - Add a **Secret**:
      - `HF_TOKEN`: Your Hugging Face token (if the model is private).

---
*Created for MLTech1 Summative Project*
