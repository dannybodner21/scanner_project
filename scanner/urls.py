from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('setup-scheduler/', views.setup_scheduler, name='setup-scheduler'),
]
