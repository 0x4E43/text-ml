from django.http import HttpResponse

def CustomView(request):
    return HttpResponse("<h1>Hello World</h1>")
