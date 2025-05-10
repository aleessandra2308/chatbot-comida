from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForCausalLM
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde .env
load_dotenv()

# Obtener credenciales desde variables de entorno
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

# Inicializar la app Flask
app = Flask(__name__)

# Ruta al modelo entrenado
MODEL_PATH = "../models/llama3_finetuned"
print(f"🔄 Cargando modelo desde: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

@app.route('/whatsapp', methods=['POST'])
def whatsapp_webhook():
    incoming_msg = request.values.get('Body', '').strip()
    print("📥 Mensaje recibido:", incoming_msg)

    # Preparar el prompt para el modelo
    input_text = f"### Prompt:\n{incoming_msg}\n\n### Response:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()

    print("🤖 Respuesta:", respuesta)

    # Crear la respuesta para Twilio
    twilio_response = MessagingResponse()
    twilio_response.message(respuesta)
    return str(twilio_response)

if __name__ == '__main__':
    print("🚀 Servidor corriendo en http://localhost:5000/whatsapp")
    app.run(host='0.0.0.0', port=5000)
