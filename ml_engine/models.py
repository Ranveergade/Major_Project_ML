from django.db import models
from django.contrib.auth.models import User
from datetime import datetime
import os

# Create your models here.
def user_dataset_path(instance, filename):
    username = instance.user.username
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{instance.user.username}/datasets/{timestamp}_{filename}"


class UploadedDataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to=user_dataset_path)
    original_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.original_name