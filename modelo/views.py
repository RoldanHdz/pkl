import os
from django.shortcuts import render
import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np

# Ruta al archivo del modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'xgboost_model_roldan.pkl')

# Cargar el modelo entrenado al iniciar el servidor
model = joblib.load(MODEL_PATH)

# Diccionario de causas atribuidas al conductor
causa_atrib_conductor = {
    16: 'VELOCIDAD INMODERADA',
    10: 'NO RESPETAR CEDA EL PASO',
    14: 'OTRO',
    5: 'CAMBIAR CARRIL SIN PRECAUCIÓN',
    9: 'NO GUARDAR DISTANCIA',
    12: 'NO RESPETAR SEMÁFORO',
    4: 'CAMBIAR CARRIL SIN PRECAUCION',
    17: 'VUELTA EN "U"',
    15: 'REBASAR INDEBIDAMENTE',
    13: 'NO RESPETAR SEÑAL DE ALTO',
    11: 'NO RESPETAR SEMAFORO',
    2: 'ALCOHOL / DROGAS',
    6: 'CIRCULAR EN SENTIDO CONTRARIO',
    3: 'ALCOHOL/DROGAS',
    0: 'ABRIR PUERTA SIN PRECAUCION',
    8: 'EXCESO DE DIMENSIONES',
    1: 'ABRIR PUERTA SIN PRECAUCIÓN',
    7: 'nan'
}

@csrf_exempt
def prediction_form(request):
    return render(request, 'predict_form.html') 

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

                # Obtener el texto correspondiente a la predicción
                causa = causa_atrib_conductor.get(prediction_list[0], "Desconocido")

                # Retornar la predicción y su causa correspondiente como respuesta JSON
                return JsonResponse({'prediction': prediction_list, 'causa': causa})

            else:
                return JsonResponse({'error': 'No features provided'}, status=400)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except ValueError as ve:
            return JsonResponse({'error': str(ve)}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

def test_model(request):
    return JsonResponse({'message': 'Modelo cargado correctamente'})
