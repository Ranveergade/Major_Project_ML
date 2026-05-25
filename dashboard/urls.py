# dashboard/urls.py
from django.urls import path
from django.http import HttpResponse

def placeholder(request):
    return HttpResponse("Dashboard Coming Soon!")

app_name = 'dashboard'

urlpatterns = [
    path('', placeholder, name='dashboard'),
]