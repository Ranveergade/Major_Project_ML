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
from django.contrib.auth.decorators import login_required
from .models import UploadedDataset,User
from .ml_core import MLEngine



# Store file paths in memory (more reliable than session)
_file_path_store = {}

@csrf_exempt

def upload_file(request):
    context = {}
    if request.user.is_authenticated:
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
                    
                
                existing_file=UploadedDataset.objects.filter(
                    user=request.user,
                    original_name=uploaded_file.name
                ).first()
                if existing_file:
                    context["error"]= f"{uploaded_file.name} already uploaded. Please check Results Page for ML analysis."
                    

                dataset_obj = UploadedDataset.objects.create(
                        user=request.user,
                        file=uploaded_file,
                        original_name=uploaded_file.name
                                                            )
                
                full_path = dataset_obj.file.path


                if file_ext == "csv":
                    df = pd.read_csv(full_path)
                elif file_ext in ["xlsx", "xls"]:
                    df = pd.read_excel(full_path)
                else:
                    df = pd.read_json(full_path)

                ml = MLEngine()

                ml.df = df

                detection = ml.auto_detect_dataset_type()

                

                request.session["uploaded_file_path"] = full_path
                request.session["uploaded_filename"] = uploaded_file.name
                request.session.modified = True

                context["success"]= True
                context["filename"]=uploaded_file.name
                context["rows"]=df.shape[0]
                context["cols"]=df.shape[1]
                context["columns"]=[str(col) for col in df.columns ]
                context["problem_type"] = detection["type"]

            except Exception as e:
                import traceback
                print(traceback.format_exc())
                context["error"] = str(e)
            
            print("FINAL CONTEXT =", context)

        return render(request, "ml_engine/upload.html", context)
    else:
        return redirect("accounts:login")

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

@login_required
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

                if "classification_report" in model_data:

                    model_data["macro_avg"] = model_data["classification_report"]["macro avg"]

                    model_data["macro_avg"]["f1_score"] = model_data["macro_avg"].pop("f1-score")


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
@login_required
def results_page(request):
    ml_results = request.session.get('ml_results', {})
    return render(request, 'ml_engine/results.html', {'results': ml_results})



@login_required
def upload_history(request):
    files = UploadedDataset.objects.filter(user=request.user).order_by("-uploaded_at")
    return render(request, "ml_engine/history.html", {"files": files})

@login_required
def run_unsupervised(request):
    

    if request.method != "POST":
        return redirect("upload")


    file_path = request.session.get("uploaded_file_path")


    if not file_path:
        return render(request,"ml_engine/upload.html",{
            "error":"No file uploaded"
        })
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
            df = pd.read_csv(file_path)

    elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)

    elif ext == ".json":
            df = pd.read_json(file_path)

    else:
            return render(request, "ml_engine/upload.html", {
                "error": "Unsupported file type"
            })


    try:

        from .ml_core import MLEngine


        ml = MLEngine()


        ml.load_data(file_path)

        ml.clean_data()


        algorithms = [
            "K-Means",
            "DBSCAN",
            "Hierarchical",
            "PCA",
            "t-SNE",
            "Isolation Forest",
            "Local Outlier Factor"
        ]


        all_results = {}


        for algo in algorithms:

            result = ml.train_unsupervised(
                algorithm_name=algo
            )

            all_results[algo] = result
        visualizations = ml.create_unsupervised_visualizations(
        algorithm_name=ml.best_model
)
        



        ml_results = {

            "problem_type":"unsupervised",

            "all_models": make_json_safe(all_results),

            "best_model": ml.best_model,

            "best_score": ml.best_score,

            "visualizations":visualizations

        }



        request.session["ml_results"] = ml_results



        return render(
            request,
            "ml_engine/results.html",
            {
                "results":ml_results
            }
        )


    except Exception as e:

        print("UNSUPERVISED ERROR:", str(e))
        print(traceback.format_exc())


        return render(
            request,
            "ml_engine/upload.html",
            {
                "error":str(e)
            }
        )