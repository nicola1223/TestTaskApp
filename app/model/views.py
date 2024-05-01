from django.shortcuts import render


def index(request):
    context = {
        'model': 'Gradient Boosted Trees'
    }
    return render(request, template_name="model/index.html", context=context)

