from unsloth import FastLanguageModel
import torch, json
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Paso 1: Cargar modelo base
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3-8B-Instruct",
    max_seq_length = 2048,
    dtype = torch.float16,          # o None si no tienes soporte para float16
    load_in_4bit = True,
)

# Paso 2: Configurar LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0.1,
    bias = "none",
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "down_proj", "up_proj"
    ],
)

# Paso 3: Cargar dataset local
with open("data/processed/dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Paso 4: Formatear al estilo Alpaca
def format_example(example):
    return {
        "text": f"### Prompt:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

formatted_data = [format_example(e) for e in raw_data]
dataset = Dataset.from_list(formatted_data)

# Paso 5: Definir parámetros de entrenamiento
training_args = TrainingArguments(
    output_dir="models/llama3_finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    optim="paged_adamw_8bit",
    lr_scheduler_type="linear",
    warmup_steps=10,
    save_strategy="epoch",
    seed=42,
    report_to="none",
)

# Paso 6: Preparar trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
    args=training_args,
)

# Paso 7: Entrenar
trainer.train()

# Paso 8: Guardar modelo y tokenizer
model.save_pretrained("models/llama3_finetuned")
tokenizer.save_pretrained("models/llama3_finetuned")

