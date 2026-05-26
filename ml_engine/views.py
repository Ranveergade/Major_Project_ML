# ml_engine/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import pandas as pd
import os

def upload_file(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        
        if uploaded_file:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            allowed_ext = ['csv', 'xlsx', 'json', 'xls']
            
            if file_ext not in allowed_ext:
                return JsonResponse({'error': 'Invalid file type'}, status=400)
            
            file_path = default_storage.save(uploaded_file.name, ContentFile(uploaded_file.read()))
            file_size = uploaded_file.size
            
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
                
                request.session['uploaded_file_path'] = full_path
                request.session['uploaded_filename'] = uploaded_file.name
                
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

def run_analysis(request):
    if request.method == 'POST':
        target_column = request.POST.get('target_column')
        uploaded_file = request.FILES.get('file')
        
        if not target_column:
            return JsonResponse({'error': 'Please select target column'}, status=400)
        
        file_path = request.session.get('uploaded_file_path')
        
        if not file_path:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        
        try:
            from .ml_core import MLEngine
            
            ml = MLEngine()
            df = ml.load_data(file_path)
            problem_type = ml.detect_problem_type(target_column)
            ml.clean_data()
            results = ml.train_models()
            
            request.session['ml_results'] = {
                'problem_type': problem_type,
                'target_column': target_column,
                'best_model': ml.best_model,
                'best_score': ml.best_score,
                'all_models': results
            }
            
            return JsonResponse({
                'success': True,
                'problem_type': problem_type,
                'best_model': ml.best_model,
                'best_score': ml.best_score,
                'all_results': {k: v for k, v in results.items() if 'error' not in v}
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def results_page(request):
    """Results page view"""
    ml_results = request.session.get('ml_results', {})
    return render(request, 'ml_engine/results.html', {'results': ml_results})