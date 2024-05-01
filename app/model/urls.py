from django.urls import path
from model.views import index

urlpatterns = [
    path('', index)
]
