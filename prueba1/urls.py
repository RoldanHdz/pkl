from django.contrib import admin
from django.urls import path
from modelo.views import predict, test_model, prediction_form


urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', predict, name='predict'),  # Ruta para predicciones
    path('test/', test_model, name='test'),  # Ruta para probar el modelo
    path('test_model/', test_model, name='test_model'),
    path('prediction/', prediction_form, name='prediction_form')

]
