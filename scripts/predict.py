from transformers import AutoTokenizer, AutoModelForCausalLM

# Ruta del modelo entrenado
MODEL_PATH = "models/llama3_finetuned"

# Cargar el modelo y el tokenizer
print(f"🔄 Cargando modelo desde: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

def generar_respuesta(prompt):
    input_text = f"### Prompt:\n{prompt}\n\n### Response:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()

# Prueba directa
if __name__ == "__main__":
    while True:
        prompt = input("🧠 Escribe tu pregunta (o 'salir'): ")
        if prompt.lower() == "salir":
            break
        respuesta = generar_respuesta(prompt)
        print("🤖 Respuesta del modelo:", respuesta)
