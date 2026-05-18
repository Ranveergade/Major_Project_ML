from django.shortcuts import render
from django.http import HttpResponse  
from django import forms
import pandas as pd
from PyPDF2 import PdfReader  
# Create your views here.                                                                                                           
class FileUploadForm(forms.Form):
    file = forms.FileField()

def placeholder(request):
    return HttpResponse("ML Engine - Coming Soon!")
#file extension understanding
def file_seeing(request):

    result = None

    if request.method == "POST":

        form = FileUploadForm(request.POST, request.FILES)

        if form.is_valid():

            uploaded_file = request.FILES["file"]

            filename = uploaded_file.name
            extension = filename.split(".")[-1].lower()

            try:

                # CSV FILE
                if extension == "csv":

                    df = pd.read_csv(uploaded_file)

                    result = {
                        "success": True,
                        "type": "table",
                        "filename": filename,
                        "columns": list(df.columns),
                        "rows": df.head(20).fillna("").values.tolist(),
                        "total_rows": df.shape[0],
                        "total_columns": df.shape[1],
                        "file_type": extension,
                    }
                # EXCEL FILE
                elif extension in ["xlsx", "xls"]:

                    df = pd.read_excel(uploaded_file)

                    result = {
                        "success": True,
                        "type": "table",
                        "filename": filename,
                        "columns": list(df.columns),
                        "rows": df.head(20).fillna("").values.tolist(),
                        "total_rows": df.shape[0],
                        "total_columns": df.shape[1],
                        "file_type": extension,
                    }

                # PDF FILE
                elif extension == "pdf":

                    reader = PdfReader(uploaded_file)

                    text = ""

                    for page in reader.pages:

                        page_text = page.extract_text()

                        if page_text:
                            text += page_text + "\n"

                    result = {
                        "success": True,
                        "type": "text",
                        "filename": filename,
                        "content": text[:5000],
                        "total_characters": len(text),
                        "file_type": extension,
                    }

                # TXT FILE
                elif extension == "txt":

                    text = uploaded_file.read().decode(
                        "utf-8",
                        errors="ignore"
                    )

                    result = {
                        "success": True,
                        "type": "text",
                        "filename": filename,
                        "content": text[:5000],
                        "total_characters": len(text),
                        "file_type": extension, 
                    }

                else:

                    result = {
                        "success": False,
                        "error": "Unsupported File Type"
                    }

            except Exception as e:

                result = {
                    "success": False,
                    "error": str(e)
                }

    else:
        form = FileUploadForm()

    return render(
        request,
        "ml_engine/upload_file.html",
        {
            "form": form,
            "result": result
        }
    )