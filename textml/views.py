from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

def index(request):
    try:
        return render(request, 'index.html')
    except Exception as e:
        print(e)
        print("Error rendering template: {}", e)

def upload(request):
    if request.method == 'POST':
        try:
            myfile = request.FILES['file-upload']
            print("Received file:", myfile.name)
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            print("Uploaded file URL:", uploaded_file_url)
            fileurl = '.' + uploaded_file_url
            print("File URL on server:", fileurl)
        except Exception as e:
            print("Error uploading file:", e)
    else:
        print("GET request received")
    return render(request, 'index.html')

def home(request):
    return HttpResponse("home Page works")