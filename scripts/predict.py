from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "models/llama3_finetuned"
print(f"🔄 Cargando modelo desde: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to("cuda")  # 👉 envía el modelo a la GPU

def generar_respuesta(prompt):
    entrada = f"### Prompt:\n{prompt}\n\n### Response:"
    inputs = tokenizer(entrada, return_tensors="pt").to("cuda")  # 👉 envía los inputs a la GPU también
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return respuesta.replace(entrada, "").strip()

# Prueba interactiva
if __name__ == "__main__":
    while True:
        prompt = input("🧠 Escribe tu pregunta (o 'salir'): ")
        if prompt.lower() in ["salir", "exit"]:
            break
        print("🤖 Bot:", generar_respuesta(prompt))
