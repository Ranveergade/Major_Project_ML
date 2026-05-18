# ml_engine/urls.py
from django.urls import path
from . import views

app_name = 'ml_engine'

urlpatterns = [
    # Placeholder - will add upload views later
    path('', views.placeholder, name='home'),
    path("upload", views.file_seeing, name="upload")
]