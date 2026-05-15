# dashboard/urls.py
from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    # Placeholder
    path('', views.placeholder, name='dashboard'),
]