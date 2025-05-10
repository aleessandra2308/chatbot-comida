import pandas as pd
import json
import os

def convertir_csv_a_json(input_csv, output_json):
    """
    Convierte un archivo CSV con columnas 'prompt' y 'response'
    al formato JSON esperado por el script de entrenamiento.
    """
    df = pd.read_csv(input_csv)

    # Validación
    if not {'prompt', 'response'}.issubset(df.columns):
        raise ValueError("El CSV debe tener columnas 'prompt' y 'response'.")

    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "instruction": row["prompt"],
            "input": "",
            "output": row["response"]
        })

    # Asegurar que la carpeta de salida exista
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Dataset convertido y guardado en: {output_json}")
