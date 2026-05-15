from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

def home(request):
    """Home page - ML Analytics Pro"""
    context = {
        'title': 'ML Analytics Pro - AI-Powered AutoML'
    }
    return render(request, 'core/home.html', context)