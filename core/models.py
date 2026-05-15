from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    credits = models.IntegerField(default=1000)
    is_premium = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"

class Dataset(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='datasets/%Y/%m/%d/')
    filename = models.CharField(max_length=255)
    file_size = models.BigIntegerField()
    rows = models.IntegerField(null=True, blank=True)
    columns = models.IntegerField(null=True, blank=True)
    file_type = models.CharField(max_length=10, choices=[('csv', 'CSV'), ('xlsx', 'Excel'), ('json', 'JSON')])
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    task_id = models.CharField(max_length=100, blank=True, null=True)
    processing_time = models.DurationField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Dataset"
        verbose_name_plural = "Datasets"

    def __str__(self):
        return f"{self.filename} - {self.status}"

class AnalysisResult(models.Model):
    dataset = models.OneToOneField(Dataset, on_delete=models.CASCADE, related_name='analysis')
    problem_type = models.CharField(max_length=20, choices=[('regression', 'Regression'), ('classification', 'Classification')])
    best_model = models.CharField(max_length=100)
    best_score = models.FloatField()
    all_models = models.JSONField()  # {"model_name": {"accuracy": 0.95, "metrics": {...}}}
    eda_summary = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Analysis Result"
        verbose_name_plural = "Analysis Results"








