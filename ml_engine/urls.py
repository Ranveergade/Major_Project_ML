# ml_engine/urls.py
from django.urls import path
from . import views

app_name = 'ml_engine'

urlpatterns = [
    path('', views.upload_file, name='upload'),
    path('analyze/', views.run_analysis, name='analyze'),
    path('results/', views.results_page, name='results'),
]