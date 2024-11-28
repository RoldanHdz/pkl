import os
import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np

# Ruta al archivo del modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'xgboost_model_roldan.pkl')

# Cargar el modelo entrenado al iniciar el servidor
model = joblib.load(MODEL_PATH)

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            # Obtener los datos JSON enviados en el cuerpo de la solicitud
            data = json.loads(request.body)  # Convertir el cuerpo en un diccionario
            features = data.get('features')  # Obtener la lista de características

            if features:
                # Convertir las características en un arreglo NumPy
                features_array = np.array(features).reshape(1, -1)

                # Realizar la predicción
                prediction = model.predict(features_array)

                # Convertir la predicción en una lista antes de enviarla en la respuesta
                prediction_list = prediction.tolist()

                # Retornar la predicción como respuesta JSON
                return JsonResponse({'prediction': prediction_list})

            else:
                return JsonResponse({'error': 'No features provided'}, status=400)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except ValueError as ve:
            return JsonResponse({'error': str(ve)}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

def test_model(request):
    return JsonResponse({'message': 'Modelo cargado correctamente'})
