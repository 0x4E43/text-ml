from django.http import HttpResponse
from django.shortcuts import render


def index(request):
    try:
        return render(request, 'index.html')
    except Exception as e:
        print(e)
        print("Error rendering template: {}", e)

def home(request):
    return HttpResponse("home Page works")