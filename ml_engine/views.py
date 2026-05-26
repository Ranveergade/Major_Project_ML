# ml_engine/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import os
import traceback
import json

# Store file paths in memory (more reliable than session)
_file_path_store = {}

@csrf_exempt
def upload_file(request):
    if request.method != 'POST':
        return render(request, 'ml_engine/upload.html')
    
    uploaded_file = request.FILES.get('file')
    
    if not uploaded_file:
        return JsonResponse({'error': 'No file provided'}, status=400)
    
    file_ext = uploaded_file.name.split('.')[-1].lower()
    allowed_ext = ['csv', 'xlsx', 'json', 'xls']
    
    if file_ext not in allowed_ext:
        return JsonResponse({'error': 'Invalid file type'}, status=400)
    
    try:
        # Save file with unique name
        unique_name = f"dataset_{uploaded_file.name}"
        file_path = default_storage.save(unique_name, ContentFile(uploaded_file.read()))
        full_path = os.path.join('media/', file_path)
        
        # Read file info
        if file_ext == 'csv':
            df = pd.read_csv(full_path)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(full_path)
        elif file_ext == 'json':
            df = pd.read_json(full_path)
        
        rows, cols = df.shape
        columns = list(df.columns)
        
        # Store in global dict with session key
        session_key = request.session.session_key
        _file_path_store[session_key] = full_path
        
        request.session['uploaded_file_path'] = full_path
        request.session['uploaded_filename'] = uploaded_file.name
        
        return JsonResponse({
            'success': True,
            'filename': uploaded_file.name,
            'rows': rows,
            'columns': columns
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
def run_analysis(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid method'}, status=400)
    
    # Get target column
    target_column = request.POST.get('target_column')
    if not target_column:
        return JsonResponse({'error': 'Please select target column'}, status=400)
    
    # Get file path - check both session and global store
    session_key = request.session.session_key
    file_path = request.session.get('uploaded_file_path') or _file_path_store.get(session_key)
    
    if not file_path:
        return JsonResponse({'error': 'No file uploaded. Please upload a file first.'}, status=400)
    
    if not os.path.exists(file_path):
        return JsonResponse({'error': 'File not found. Please upload again.'}, status=400)
    
    try:
        from .ml_core import MLEngine
        
        ml = MLEngine()
        df = ml.load_data(file_path)
        problem_type = ml.detect_problem_type(target_column)
        ml.clean_data()
        results = ml.train_models()
        visualizations = ml.create_visualizations()
        eda_summary = ml.get_eda_summary()
        
        # Clean results for JSON serialization
        clean_results = {}
        for model_name, model_data in results.items():
            clean_results[model_name] = {k: v for k, v in model_data.items() 
                                      if k not in ['model', 'feature_importance']}
        
        # Store in session
        request.session['ml_results'] = {
            'problem_type': problem_type,
            'target_column': target_column,
            'best_model': ml.best_model,
            'best_score': ml.best_score,
            'all_models': clean_results,
            'visualizations': visualizations,
            'eda_summary': eda_summary,
            'rows': eda_summary['rows'],
            'columns': eda_summary['columns']
        }
        
        return JsonResponse({
            'success': True,
            'problem_type': problem_type,
            'best_model': ml.best_model,
            'best_score': ml.best_score,
            'all_results': clean_results,
            'visualizations': visualizations,
            'eda_summary': eda_summary
        })
            
    except Exception as e:
        error_msg = str(e)
        print("ML Error:", error_msg)
        print(traceback.format_exc())
        return JsonResponse({'error': error_msg}, status=400)

def results_page(request):
    ml_results = request.session.get('ml_results', {})
    return render(request, 'ml_engine/results.html', {'results': ml_results})