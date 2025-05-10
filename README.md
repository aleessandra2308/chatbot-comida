# chatbot-comida🍗
Chatbot diseñado para un negocio de comidas

## Estructura 🚀
- `scripts/`: código para entrenamiento e inferencia.
- `data/`: datasets crudos y preprocesados.
- `models/`: modelos fine-tuned guardados.
- `app/`: API para servir el modelo vía HTTP.

chatbot_finetuning_project/
├── app/ → API Flask para conectar con WhatsApp (Twilio)
├── data/ → Datasets crudos y procesados para entrenamiento
├── logs/ → Logs de entrenamiento o ejecución
├── models/ → Modelos guardados luego del fine-tuning
├── notebooks/ → Notebook original usado en clase
├── scripts/ → Código para entrenar, probar e importar datos
├── config.yaml → Parámetros para el entrenamiento
├── requirements.txt → Dependencias principales del proyecto
└── README.md → Esta documentación
