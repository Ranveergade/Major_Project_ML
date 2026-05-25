# ml_engine/views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import pandas as pd
import os

def upload_file(request):
    """File upload page"""
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        
        if uploaded_file:
            # Validate file type
            file_ext = uploaded_file.name.split('.')[-1].lower()
            allowed_ext = ['csv', 'xlsx', 'json', 'xls']
            
            if file_ext not in allowed_ext:
                return JsonResponse({'error': 'Invalid file type. Allowed: CSV, Excel, JSON'}, status=400)
            
            # Save file
            file_path = default_storage.save(uploaded_file.name, ContentFile(uploaded_file.read()))
            
            # Get file size
            file_size = uploaded_file.size
            
            # Try to read and get info
            try:
                full_path = os.path.join('media/', file_path)
                if file_ext == 'csv':
                    df = pd.read_csv(full_path)
                elif file_ext in ['xlsx', 'xls']:
                    df = pd.read_excel(full_path)
                elif file_ext == 'json':
                    df = pd.read_json(full_path)
                
                rows, cols = df.shape
                columns = list(df.columns)
                dtypes = df.dtypes.astype(str).to_dict()
                
                return JsonResponse({
                    'success': True,
                    'filename': uploaded_file.name,
                    'file_size': file_size,
                    'rows': rows,
                    'columns': columns,
                    'dtypes': dtypes
                })
                
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=400)
    
    return render(request, 'ml_engine/upload.html')