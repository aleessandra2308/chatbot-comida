from unsloth import FastTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import json
import os

# Cargar configuración desde config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Cargar dataset procesado
with open(config["dataset_path"], "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Inicializar modelo y tokenizer
print(f"Cargando modelo base: {config['base_model']}")
tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
model = AutoModelForCausalLM.from_pretrained(config["base_model"])

# Inicializar FastTrainer
trainer = FastTrainer(model=model, tokenizer=tokenizer)

# Entrenar modelo
print("Iniciando entrenamiento...")
trainer.train(
    dataset=dataset,
    output_dir=config["output_dir"],
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    learning_rate=config["learning_rate"],
    max_seq_length=config["max_seq_length"]
)
print("Entrenamiento finalizado.")
