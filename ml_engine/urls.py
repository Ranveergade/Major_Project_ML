from django.urls import path
from . import views

app_name = 'ml_engine'

urlpatterns = [
    path('', views.upload_file, name='upload'),
]

"""urlpatterns = [
    # Placeholder - will add upload views later
    path('', views.placeholder, name='home'),
    path("upload", views.file_seeing, name="upload")
]"""