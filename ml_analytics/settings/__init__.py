"""
Django settings project configuration.
"""
from pathlib import Path
import os

# Build paths inside the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# DEFAULT to base settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_analytics.settings.base')

from .base import *