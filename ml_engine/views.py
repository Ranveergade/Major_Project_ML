# ml_engine/views.py
from django.shortcuts import render,redirect
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import os
import numpy as np
import traceback
import json

# Store file paths in memory (more reliable than session)
_file_path_store = {}

@csrf_exempt
def upload_file(request):
    context = {}

    if request.method == "POST":
        uploaded_file = request.FILES.get("file")

        if not uploaded_file:
            context["error"] = "No file uploaded"
            return render(request, "ml_engine/upload.html", context)

        file_ext = uploaded_file.name.split(".")[-1].lower()

        if file_ext not in ["csv", "xlsx", "xls", "json"]:
            context["error"] = "Invalid file type"
            return render(request, "ml_engine/upload.html", context)

        try:
            if not request.session.session_key:
                request.session.create()

            file_path = default_storage.save(
                f"dataset_{request.session.session_key}_{uploaded_file.name}",
                uploaded_file
            )

            full_path = default_storage.path(file_path)

            if file_ext == "csv":
                df = pd.read_csv(full_path)
            elif file_ext in ["xlsx", "xls"]:
                df = pd.read_excel(full_path)
            else:
                df = pd.read_json(full_path)

            request.session["uploaded_file_path"] = full_path
            request.session["uploaded_filename"] = uploaded_file.name

            context["success"] = True
            context["filename"] = uploaded_file.name
            context["rows"] = df.shape[0]
            context["cols"] = df.shape[1]
            context["columns"] = [str(col) for col in df.columns]
            context["preview"] = df.head(10).fillna("").to_dict("records")

        except Exception as e:
            context["error"] = str(e)

    return render(request, "ml_engine/upload.html", context)

@csrf_exempt
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]

    elif isinstance(obj, tuple):
        return [make_json_safe(i) for i in obj]

    elif isinstance(obj, np.integer):
        return int(obj)

    elif isinstance(obj, np.floating):
        return float(obj)

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, pd.Series):
        return obj.to_dict()

    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")

    elif pd.isna(obj):
        return None

    return obj


def run_analysis(request):
    if request.method != "POST":
        return redirect("upload_file")  # change name if your URL name is different

    target_column = request.POST.get("target_column")

    if not target_column:
        return render(request, "ml_engine/upload.html", {
            "error": "Please select target column"
        })

    session_key = request.session.session_key
    file_path = request.session.get("uploaded_file_path") or _file_path_store.get(session_key)

    if not file_path:
        return render(request, "ml_engine/upload.html", {
            "error": "No file uploaded. Please upload a file first."
        })

    if not os.path.exists(file_path):
        return render(request, "ml_engine/upload.html", {
            "error": "File not found. Please upload again."
        })

    try:
        from .ml_core import MLEngine

        ml = MLEngine()

        df = ml.load_data(file_path)

        problem_type = ml.detect_problem_type(target_column)

        ml.clean_data()
        if len(ml.df) < 5:
            return render(request, "ml_engine/upload.html", {
        "error": "Dataset is too small after cleaning. Please upload a dataset with at least 5 valid rows."
    })

        results = ml.train_models()

        visualizations = ml.create_visualizations()

        eda_summary = ml.get_eda_summary()

        clean_results = {}

        for model_name, model_data in results.items():
            clean_results[model_name] = {}

            for k, v in model_data.items():
                if k not in ["model", "feature_importance"]:
                    clean_results[model_name][k] = make_json_safe(v)

        ml_results = {
            "problem_type": make_json_safe(problem_type),
            "target_column": target_column,
            "best_model": make_json_safe(ml.best_model),
            "best_score": make_json_safe(ml.best_score),
            "all_models": make_json_safe(clean_results),
            "visualizations": make_json_safe(visualizations),
            "eda_summary": make_json_safe(eda_summary),
            "rows": make_json_safe(eda_summary.get("rows")),
            "columns": make_json_safe(eda_summary.get("columns")),
        }

        request.session["ml_results"] = ml_results

        return render(request, "ml_engine/results.html", {
            "results": ml_results
        })

    except Exception as e:
        error_msg = str(e)
        print("ML Error:", error_msg)
        print(traceback.format_exc())

        return render(request, "ml_engine/upload.html", {
            "error": error_msg
        })

def results_page(request):
    ml_results = request.session.get('ml_results', {})
    return render(request, 'ml_engine/results.html', {'results': ml_results})